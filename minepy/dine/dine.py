#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffeomorphic Information Neural Estimation

Information:
    https://doi.org/10.1609/aaai.v37i6.25908
    https://github.com/baosws/DINE/tree/main

2024-10-31
"""

import numpy as np
import schedulefree
import torch
from scipy.stats import norm
from torch import nn
from tqdm import tqdm

from minepy.dine.dine_tools import UniformFlow, data_loader
from minepy.minepy_tools import EarlyStopping

EPS = 1e-6


class DineModel(nn.Module):
    def __init__(self, dx, dy, dz, n_components, hidden_sizes):
        super().__init__()
        self.x_cnf = UniformFlow(
            d=dx, dz=dz, dh=hidden_sizes, n_components=n_components
        )
        self.y_cnf = UniformFlow(
            d=dy, dz=dz, dh=hidden_sizes, n_components=n_components
        )

    def forward(self, X, Y, Z):
        _, log_px = self.x_cnf(X, Z)
        _, log_py = self.y_cnf(Y, Z)

        loss = -torch.mean(log_px + log_py)
        return loss

    def transform(self, X, Y, Z):
        self.eval()
        ex, _ = self.x_cnf(X, Z)
        ey, _ = self.y_cnf(Y, Z)

        ex = ex.detach().cpu().numpy()
        ey = ey.detach().cpu().numpy()
        return ex, ey

    def cmi(self, X, Y, Z):
        e_x, e_y = self.transform(X, Y, Z)
        e_x, e_y = map(lambda x: np.clip(x, EPS, 1 - EPS), (e_x, e_y))
        e_x, e_y = map(norm.ppf, (e_x, e_y))

        cov_x, cov_y = map(np.cov, (e_x.T, e_y.T))
        cov_x = cov_x.reshape(e_x.shape[1], e_x.shape[1])
        cov_y = cov_y.reshape(e_y.shape[1], e_y.shape[1])
        cov_all = np.cov(np.column_stack((e_x, e_y)).T)
        cmi = 0.5 * (
            np.log(np.linalg.det(cov_x))
            + np.log(np.linalg.det(cov_y))
            - np.log(np.linalg.det(cov_all))
        )
        return cmi


class Dine(nn.Module):
    def __init__(self, X, Y, Z, n_components=16, hidden_sizes=4, device=None):

        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)

        N = X.shape[0]
        X, Y, Z = map(lambda x: (x - np.mean(x)) / np.std(x), (X, Y, Z))
        X, Y, Z = map(lambda x: torch.tensor(x).float().view(N, -1), (X, Y, Z))

        self.X = X.to(self.device)
        self.Y = Y.to(self.device)
        self.Z = Z.to(self.device)
        dx = X.shape[1]
        dy = Y.shape[1]
        dz = Z.shape[1]
        self.model = DineModel(
            dx=dx, dy=dy, dz=dz, n_components=n_components, hidden_sizes=hidden_sizes
        )
        self.model = self.model.to(self.device)

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-3,
        stop_patience=100,
        stop_min_delta=0,
        val_size=0.2,
        verbose=False,
    ):

        opt = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=lr)

        early_stopping = EarlyStopping(patience=stop_patience, delta=stop_min_delta)

        Xtrain, Ytrain, Ztrain, Xval, Yval, Zval = data_loader(
            self.X, self.Y, self.Z, val_size=val_size, device=self.device
        )

        val_loss_epoch = []
        cmi_epoch = []

        for _ in tqdm(range(max_epochs), disable=not verbose):
            rand_perm = torch.randperm(Xtrain.shape[0])
            if batch_size == "full":
                batch_size = Xtrain.shape[0]
            self.train()
            opt.train()
            for inds in rand_perm.split(batch_size, dim=0):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    train_loss = self.model(
                        Xtrain[inds, :], Ytrain[inds, :], Ztrain[inds, :]
                    )
                    train_loss.backward()
                    opt.step()

            # validate and testing
            self.eval()
            opt.eval()
            with torch.set_grad_enabled(False):
                # validate
                val_loss = self.model(Xval, Yval, Zval)
                val_loss_epoch.append(val_loss.item())
                cmi_epoch.append(self.model.cmi(self.X, self.Y, self.Z))

                early_stopping(val_loss)  # early stopping
            if early_stopping.early_stop:
                break

        self.val_loss_epoch = np.array(val_loss_epoch)
        self.cmi_epoch = np.array(cmi_epoch)

    def get_cmi(self):
        return self.cmi_epoch[np.argmin(self.val_loss_epoch)]

    def get_curves(self):
        return self.val_loss_epoch, self.cmi_epoch
