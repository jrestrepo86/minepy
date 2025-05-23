#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
import schedulefree


from minepy.mine.mine_tools import mine_data_loader
from minepy.minepy_tools import (
    EarlyStopping,
    ExpMovingAverageSmooth,
    get_activation_fn,
    toColVector,
)

EPS = 1e-10


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output * input.exp().detach() / (running_mean + EPS) / input.shape[0]
        )
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema
    return t_log, running_mean


class MineModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers=[64, 32, 16, 8],
        afn="gelu",
        loss="mine",
        alpha=0.01,
        regWeight=0.1,
        targetVal=0.0,
    ):
        super().__init__()

        hidden_layers = [int(hl) for hl in hidden_layers]

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_layers[0]), activation_fn()]
        for i in range(len(hidden_layers) - 1):
            seq += [
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(hidden_layers[-1], 1)]
        self.model = nn.Sequential(*seq)
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.regWeight = regWeight
        self.targetVal = targetVal

    def forward(self, x, y):
        y_marg = y[torch.randperm(y.shape[0])]

        t = self.model(torch.cat((x, y), dim=1)).mean()
        t_marg = self.model(torch.cat((x, y_marg), dim=1))

        if self.loss in ["mine"]:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha
            )
            mi = t - second_term
            loss = -mi
        elif self.loss in ["fdiv"]:
            second_term = torch.exp(t_marg - 1).mean()
            mi = t - second_term
            loss = -mi
        elif self.loss in ["remine"]:
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])
            mi = t - second_term
            loss = -mi + self.regWeight * torch.pow(second_term - self.targetVal, 2)
        else:
            # mine_biased as default
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])
            mi = t - second_term
            loss = -mi

        return loss, mi


class Mine(nn.Module):
    def __init__(
        self,
        X,
        Y,
        hidden_layers=[32, 16, 8, 4],
        afn="gelu",
        loss="mine_biased",
        alpha=0.01,
        regWeight=0.1,
        targetVal=0.0,
        device=None,
    ):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)
        # Vars
        self.X = toColVector(X.astype(np.float32))
        self.Y = toColVector(Y.astype(np.float32))
        self.dx = self.X.shape[1]
        self.dy = self.Y.shape[1]

        self.model = MineModel(
            input_dim=self.dx + self.dy,
            hidden_layers=hidden_layers,
            afn=afn,
            loss=loss,
            alpha=alpha,
            regWeight=regWeight,
            targetVal=targetVal,
        )
        self.model = self.model.to(self.device)

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-3,
        weight_decay=5e-5,
        stop_patience=100,
        stop_min_delta=0,
        val_size=0.2,
        random_sample="True",
        verbose=False,
    ):
        # opt = torch.optim.RMSprop(
        #     self.model.parameters(), lr=lr, weight_decay=weight_decay
        # )
        # scheduler = CyclicLR(opt, base_lr=lr, max_lr=1e-3, mode="triangular2")
        opt = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=lr)

        early_stopping = EarlyStopping(patience=stop_patience, delta=stop_min_delta)

        val_loss_ema_smooth = ExpMovingAverageSmooth()

        Xtrain, Ytrain, Xval, Yval, X, Y = mine_data_loader(
            self.X,
            self.Y,
            val_size=val_size,
            random_sample=random_sample,
            device=self.device,
        )

        val_loss_epoch = []
        val_ema_loss_epoch = []
        val_mi_epoch = []
        test_mi_epoch = []

        for _ in tqdm(range(max_epochs), disable=not verbose):
            # training
            rand_perm = torch.randperm(Xtrain.shape[0])
            if batch_size == "full":
                batch_size = Xtrain.shape[0]
            self.train()
            opt.train()
            for inds in rand_perm.split(batch_size, dim=0):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    train_loss, _ = self.model(Xtrain[inds, :], Ytrain[inds, :])
                    train_loss.backward()
                    opt.step()

            # validate and testing
            self.eval()
            opt.eval()
            with torch.set_grad_enabled(False):
                # validate
                val_loss, val_mi = self.model(Xval, Yval)
                val_loss_epoch.append(val_loss.item())
                val_ema_loss = val_loss_ema_smooth(val_loss)
                val_ema_loss_epoch.append(val_ema_loss.item())
                val_mi_epoch.append(val_mi.item())

                # scheduler.step() # learning rate scheduler

                early_stopping(val_ema_loss)  # early stopping

                # testing
                _, test_mi = self.model(X, Y)
                test_mi_epoch.append(test_mi.item())

            if early_stopping.early_stop:
                break

        self.val_loss_epoch = np.array(val_loss_epoch)
        self.val_mi_epoch = np.array(val_mi_epoch)
        self.val_ema_loss_epoch = np.array(val_ema_loss_epoch)
        self.test_mi_epoch = np.array(test_mi_epoch)

    def get_mi(self, all=False):
        ind_min_ema_loss = np.argmin(self.val_ema_loss_epoch)
        mi_val = self.val_mi_epoch[ind_min_ema_loss]
        mi_test = self.test_mi_epoch[ind_min_ema_loss]
        fepoch = self.val_ema_loss_epoch.size
        if all:
            return mi_val, mi_test, ind_min_ema_loss, fepoch
        else:
            return mi_test

    def get_curves(self):
        return (
            self.val_loss_epoch,
            self.val_ema_loss_epoch,
            self.val_mi_epoch,
            self.test_mi_epoch,
        )
