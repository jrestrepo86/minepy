#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from minepy.class_mi.class_mi_tools import ClassMiModel, class_mi_batch
from minepy.minepy_tools import EarlyStopping, get_activation_fn, toColVector

EPS = 1e-6


class ClassMI(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim=50,
                 num_hidden_layers=2,
                 afn='elu',
                 device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        torch.device(self.device)

        self.model = ClassMiModel(input_dim, hidden_dim, num_hidden_layers,
                                  afn)

        self.model = self.model.to(self.device)

    def forward(self, x, z):
        return self.model(x, z)

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

        early_stopping = EarlyStopping(patience=stop_patience,
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

        loss_fn = nn.BCEWithLogitsLoss()
        train_dk = []
        val_dk = []

        for i in tqdm(range(max_epochs), disable=not verbose):
            # training
            self.train()
            for x, z in class_mi_batch(Xtrain, Ztrain, batch_size=batch_size):
                x = x.to(self.device)
                z = z.to(self.device)
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    logits, labels, _ = self.forward(x, z)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    opt.step()

            # validate and testing
            torch.set_grad_enabled(False)
            self.eval()
            with torch.no_grad():
                train_dk.append(self.calc_mi_fn(Xtrain, Ztrain))
                val_dk.append(self.calc_mi_fn(Xval, Zval))

        self.train_dk = np.array(train_dk)
        self.val_dk = np.array(val_dk)
        pass

    def calc_mi_fn(self, x, z):
        if isinstance(x, np.ndarray):
            x = toColVector(x)
            x = torch.from_numpy(x.copy()).float().to(self.device)
        if isinstance(z, np.ndarray):
            z = toColVector(z)
            z = torch.from_numpy(z.copy()).float().to(self.device)

        _, labels, probs = self.forward(x, z)
        labels = labels[:, 0] > 0
        likelihood = probs[:, 0] / probs[:, 1]
        Dkl = torch.log(likelihood[labels]).mean() - torch.log(
            likelihood[torch.logical_not(labels)].mean())
        return Dkl.item()

    def get_curves(self):
        return self.train_dk, self.val_dk
