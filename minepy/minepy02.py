#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from minepy.mineLayers import get_activation_fn
from minepy.mineTools import EarlyStopping02, MIbatch, toColVector

EPS = 1e-6


class EMALoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
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


class Mine(nn.Module):

    def __init__(self,
                 input_dim,
                 loss='mine_biased',
                 afn='relu',
                 hidden_dim=50,
                 nLayers=2,
                 alpha=0.1,
                 regWeight=1.0,
                 targetVal=0.0,
                 device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        torch.device(self.device)

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(nLayers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*seq)
        self.model = self.model.to(self.device)
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.regWeight = regWeight
        self.targetVal = targetVal
        self.calc_curves = False

    def forward(self, x, z):
        z_marg = z[torch.randperm(z.shape[0])]

        t = self.model(torch.cat((x, z), dim=1)).mean()
        t_marg = self.model(torch.cat((x, z_marg), dim=1))
        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg,
                                                      self.running_mean,
                                                      self.alpha)

            mi = t - second_term
            loss = -mi
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
            mi = t - second_term
            loss = -mi
        elif self.loss in ["remine"]:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])  # mine_biased as default

            mi = t - second_term
            loss = -mi + self.regWeight * torch.pow(
                second_term - self.targetVal, 2)
        else:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])  # mine_biased as default
            mi = t - second_term
            loss = -mi

        return loss, mi

    def fit(
        self,
        X,
        Z,
        batch_size=64,
        max_epochs=2000,
        val_size=0.2,
        lr=1e-3,
        lr_factor=0.1,
        lr_patience=10,
        stop_patience=100,
        stop_min_delta=0.05,
        verbose=False,
    ):

        opt = torch.optim.Adam(self.model.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(opt,
                                      mode='min',
                                      factor=lr_factor,
                                      patience=lr_patience,
                                      verbose=verbose)

        early_stopping = EarlyStopping02(patience=stop_patience,
                                         delta=stop_min_delta)

        X = torch.from_numpy(toColVector(X.astype(np.float32)))
        Z = torch.from_numpy(toColVector(Z.astype(np.float32)))

        N, _ = X.shape

        val_size = int(val_size * N)
        inds = np.random.permutation(N)
        val_idx, train_idx, = inds[:val_size], inds[val_size:]
        Xval, Xtrain = X[val_idx, :], X[train_idx, :]
        Zval, Ztrain = Z[val_idx, :], Z[train_idx, :]

        Xval = Xval.to(self.device)
        Xtrain = Xtrain.to(self.device)
        Zval = Zval.to(self.device)
        Ztrain = Ztrain.to(self.device)
        X = X.to(self.device)
        Z = Z.to(self.device)

        train_loss_epoch = []
        train_mi_epoch = []
        val_loss_epoch = []
        val_ema_loss_epoch = []
        val_mi_epoch = []
        test_loss_epoch = []
        test_mi_epoch = []

        for i in tqdm(range(max_epochs), disable=not verbose):
            # training
            self.train()
            for x, z in MIbatch(Xtrain, Ztrain, batch_size=batch_size):
                x = x.to(self.device)
                z = z.to(self.device)
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    train_loss, train_mi = self.forward(x, z)
                    train_loss.backward()
                    opt.step()
            train_loss, train_mi = self.forward(Xtrain, Ztrain)
            train_loss_epoch.append(train_loss.item())
            train_mi_epoch.append(train_mi.item())

            # validate and testing
            torch.set_grad_enabled(False)
            self.eval()
            with torch.no_grad():
                # validate
                val_loss, val_mi = self.forward(Xval, Zval)
                if i == 0:
                    val_ema_loss = val_loss
                else:
                    val_ema_loss = ema(val_loss, 0.01, val_ema_loss)
                val_ema_loss_epoch.append(val_ema_loss.item())
                val_loss_epoch.append(val_loss.item())
                val_mi_epoch.append(val_mi.item())

                # testing
                test_loss, test_mi = self.forward(X, Z)
                test_loss_epoch.append(test_loss.item())
                test_mi_epoch.append(test_mi.item())

                # learning rate scheduler
                scheduler.step(val_ema_loss)
                # early stopping
                early_stopping(val_ema_loss)

            if early_stopping.early_stop:
                break

        self.train_loss_epoch = np.array(train_loss_epoch)
        self.train_mi_epoch = np.array(train_mi_epoch)
        self.val_loss_epoch = np.array(val_loss_epoch)
        self.val_mi_epoch = np.array(val_mi_epoch)
        self.test_mi_epoch = np.array(test_mi_epoch)
        self.val_ema_loss_epoch = np.array(val_ema_loss_epoch)
        self.val_ema_mi_epoch = -self.val_ema_loss_epoch

    def get_mi(self):
        ind_max_stop = np.argmax(self.val_ema_mi_epoch)
        ind_min_stop = np.argmin(self.val_ema_loss_epoch)
        ind_max_val = np.argmax(self.val_mi_epoch)
        mi_val = self.val_mi_epoch[ind_max_val]
        mi_test = self.test_mi_epoch[ind_max_val]
        mi_stop = self.test_mi_epoch[ind_max_stop]
        fepoch = self.val_mi_epoch.size
        return (mi_val, mi_test, mi_stop, ind_max_val, ind_min_stop, fepoch)

    def get_curves(self):
        return (self.train_loss_epoch, self.val_loss_epoch, self.test_mi_epoch,
                self.val_ema_loss_epoch)
        # return (self.train_mi_epoch, self.val_mi_epoch, self.test_mi_epoch,
        #         self.val_ema_mi_epoch)
