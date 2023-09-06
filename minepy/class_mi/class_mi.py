#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier based mutual information

@inproceedings{
  title={CCMI: Classifier based conditional mutual information estimation},
  author={Mukherjee, Sudipto and Asnani, Himanshu and Kannan, Sreeram},
  booktitle={Uncertainty in artificial intelligence},
  pages={1083--1093},
  year={2020},
  organization={PMLR}
}
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from minepy.class_mi.class_mi_tools import batch, class_mi_data_loader
from minepy.minepy_tools import (EarlyStopping, ExpMovingAverageSmooth,
                                 get_activation_fn, toColVector)

EPS = 1e-10


class ClassMiModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=50, num_hidden_layers=2, afn="elu", device=None
    ):
        super().__init__()

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(num_hidden_layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 2)]
        self.net = nn.Sequential(*seq).to(device)
        self.softm = nn.Softmax(dim=1)

    def forward(self, samples):
        logits = torch.squeeze(self.net(samples))
        return logits

    def calc_mi_fn(self, samples, labels):
        logit = self.forward(samples)
        probs = self.softm(logit)
        inds = torch.argmax(labels, dim=1) > 0
        marg_inds, joint_inds = inds, torch.logical_not(inds)

        # get loss function
        loss = self.loss_fn(logit, labels)

        # calculate dkl bound
        joint_gamma = probs[joint_inds, 0]
        marg_gamma = probs[marg_inds, 0]

        likel_ratio_p = (joint_gamma + EPS) / (1 - joint_gamma - EPS)
        likel_ratio_q = (marg_gamma + EPS) / (1 - marg_gamma - EPS)

        fp = torch.log(torch.abs(likel_ratio_p))
        fq = torch.log(torch.abs(likel_ratio_q))

        Dkl = fp.mean() - (torch.logsumexp(fq, 0) - math.log(fq.shape[0]))
        return Dkl, loss

    def fit_model(
        self,
        train_samples,
        train_labels,
        val_samples,
        val_labels,
        batch_size=64,
        max_epochs=2000,
        lr=1e-4,
        lr_factor=0.1,
        lr_patience=10,
        stop_patience=100,
        stop_min_delta=0,
        weight_decay=5e-5,
        verbose=False,
    ):
        # opt = torch.optim.Adam(
        #     self.net.parameters(),
        #     lr=lr,
        #     weight_decay=weight_decay,
        #     betas=(0.9, 0.999)
        # )

        opt = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(
            opt, mode="min", factor=lr_factor, patience=lr_patience, verbose=verbose
        )

        early_stopping = EarlyStopping(
            patience=stop_patience, delta=int(stop_min_delta)
        )

        val_loss_ema_smooth = ExpMovingAverageSmooth()

        self.loss_fn = nn.CrossEntropyLoss()
        val_loss_epoch, val_loss_smooth_epoch, val_dkl_epoch = [], [], []

        for i in tqdm(range(max_epochs), disable=not verbose):
            # training
            rand_perm = torch.randperm(train_samples.shape[0])
            if batch_size == "full":
                batch_size = train_samples.shape[0]
            self.train()
            for inds in rand_perm.split(batch_size, dim=0):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    logits = self.forward(train_samples[inds, :])
                    loss = self.loss_fn(logits, train_labels[inds, :])
                    loss.backward()
                    opt.step()

            # validate and testing
            self.eval()
            with torch.set_grad_enabled(False):
                dkl, val_loss = self.calc_mi_fn(val_samples, val_labels)
                val_dkl_epoch.append(dkl.item())
                val_loss_epoch.append(val_loss.item())
                # val_loss = loss
                val_loss_smooth = val_loss_ema_smooth(val_loss)
                val_loss_smooth_epoch.append(val_loss_smooth.item())
                # learning rate scheduler
                scheduler.step(val_loss)
                # early stopping
                early_stopping(val_loss_smooth)

            if early_stopping.early_stop:
                break

        fepoch = i
        return (
            np.array(val_dkl_epoch),
            np.array(val_loss_epoch),
            fepoch,
        )


class ClassMI(nn.Module):
    def __init__(
        self, X, Y, hidden_dim=50, num_hidden_layers=2, afn="elu", device=None
    ):
        super().__init__()
        # select device
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

        # setup model
        self.model = ClassMiModel(
            input_dim=self.dx + self.dy,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            afn=afn,
            device=self.device,
        )

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        val_size=0.2,
        lr=1e-4,
        lr_factor=0.1,
        lr_patience=10,
        stop_patience=100,
        stop_min_delta=0.05,
        weight_decay=5e-5,
        verbose=False,
    ):
        fit_params = {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "lr": lr,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "stop_patience": stop_patience,
            "stop_min_delta": stop_min_delta,
            "weight_decay": weight_decay,
            "verbose": verbose,
        }

        self.data_loader = class_mi_data_loader(
            self.X, self.Y, val_size=val_size, device=self.device
        )

        (
            self.val_dkl_epoch,
            self.val_loss_epoch,
            self.fepoch,
        ) = self.model.fit_model(
            self.data_loader.train_samples,
            self.data_loader.train_labels,
            self.data_loader.val_samples,
            self.data_loader.val_labels,
            **fit_params
        )

    def get_mi(self, all=False):
        ind_min_loss = np.argmin(self.val_loss_epoch)
        mi_val_loss = self.val_dkl_epoch[ind_min_loss]
        mi_dkl_max = self.val_dkl_epoch.max()
        if all:
            return mi_dkl_max, mi_val_loss, self.fepoch
        else:
            return mi_val_loss

    def get_curves(self):
        return (
            self.val_dkl_epoch,
            self.val_loss_epoch,
        )
