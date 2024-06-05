#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import schedulefree

from minepy.mi_nee.mi_nee_tools import hnee_data_loader
from minepy.minepy_tools import (
    EarlyStopping,
    ExpMovingAverageSmooth,
    get_activation_fn,
    toColVector,
)

EPS = 1e-10


class HNeeModel(nn.Module):
    """docstring for ."""

    def __init__(
        self,
        input_dim,
        hidden_layers=[150, 150],
        afn="gelu",
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
        self.net = nn.Sequential(*seq)

    def forward(self, x, xref):
        mean_f = self.net(x).mean()
        log_mean_ef_ref = torch.logsumexp(self.net(xref), 0) - math.log(xref.shape[0])
        div = mean_f - log_mean_ef_ref
        return -div


class HNee(nn.Module):
    def __init__(
        self,
        X,
        hidden_layers=[32, 16, 8, 4],
        afn="gelu",
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
        self.dx = self.X.shape[1]
        self.model = HNeeModel(
            input_dim=self.dx,
            hidden_layers=hidden_layers,
            afn=afn,
        )
        self.model = self.model.to(self.device)

        self.ref_min = torch.tensor(X.min()).to(device)
        self.ref_max = torch.tensor(X.max()).to(device)
        self.href = torch.tensor(math.log(self.ref_max - self.ref_min)).to(device)

    def ref_sample_(self, x):
        ref_min, ref_max = self.ref_min, self.ref_max
        ref_samp = torch.rand((self.n_ref_samples, x.shape[1]), device=self.device)
        return (ref_max - ref_min) * ref_samp + ref_min

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-3,
        ref_batch_factor=4,
        stop_patience=500,
        stop_min_delta=0,
        val_size=0.2,
        verbose=False,
    ):

        opt = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=stop_patience, delta=stop_min_delta)
        val_loss_ema_smooth = ExpMovingAverageSmooth()

        Xtrain, Xval, X = hnee_data_loader(
            self.X, val_size=val_size, device=self.device
        )

        if batch_size == "full":
            batch_size = Xtrain.shape[0]
        self.n_ref_samples = int(batch_size * ref_batch_factor)

        val_loss_epoch = []
        val_ema_loss_epoch = []
        val_h_epoch = []
        test_h_epoch = []

        for _ in tqdm(range(max_epochs), disable=not verbose):
            # training
            rand_perm = torch.randperm(Xtrain.shape[0])
            self.train()
            opt.train()
            for inds in rand_perm.split(batch_size, dim=0):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    ref_samp = self.ref_sample_(Xtrain[inds, :])
                    train_loss = self.model(Xtrain[inds, :], ref_samp)
                    train_loss.backward()
                    opt.step()

            # validate and testing
            self.eval()
            opt.eval()
            with torch.set_grad_enabled(False):
                # validate
                ref_samp = self.ref_sample_(Xval)
                val_loss = self.model(Xval, ref_samp)
                val_loss_epoch.append(val_loss.item())
                val_ema_loss = val_loss_ema_smooth(val_loss)
                val_ema_loss_epoch.append(val_ema_loss.item())

                h_ref = self.href + val_loss
                val_h_epoch.append(h_ref.item())

                early_stopping(val_ema_loss)  # early stopping

                # testing
                ref_samp = self.ref_sample_(X)
                test_loss = self.model(X, ref_samp)
                h_ref = self.href + test_loss
                test_h_epoch.append(h_ref.item())

            if early_stopping.early_stop:
                break

        self.val_loss_epoch = np.array(val_loss_epoch)
        self.val_h_epoch = np.array(val_h_epoch)
        self.val_ema_loss_epoch = np.array(val_ema_loss_epoch)
        self.test_h_epoch = np.array(test_h_epoch)

    def get_h(self, all=False):
        ind_min_ema_loss = np.argmin(self.val_ema_loss_epoch)
        h_val = self.val_h_epoch[ind_min_ema_loss]
        h_test = self.test_h_epoch[ind_min_ema_loss]
        fepoch = self.val_ema_loss_epoch.size
        if all:
            return h_val, h_test, ind_min_ema_loss, fepoch
        else:
            return h_test

    def get_curves(self):
        return (
            self.val_loss_epoch,
            self.val_ema_loss_epoch,
            self.val_h_epoch,
            self.test_h_epoch,
        )
