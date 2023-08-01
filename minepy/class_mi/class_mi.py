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

from minepy.class_mi.class_mi_tools import ClassMiModel, class_mi_data_loader
from minepy.minepy_tools import EarlyStopping, toColVector

EPS = 1e-6


class ClassMI(nn.Module):
    def __init__(
        self, X, Y, Z=None, hidden_dim=50, num_hidden_layers=2, afn="elu", device=None
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
        if Z is None:
            self.dz = 0
        else:
            self.Z = toColVector(Z.astype(np.float32))
            self.dz = self.Z.shape[1]

        # setup model
        self.model = ClassMiModel(
            input_dim=self.dx + self.dy + self.dz,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            afn=afn,
        )
        self.model = self.model.to(self.device)

    def forward(self, samples):
        # evaluate model
        logit, probs = self.model(samples)
        return logit, probs

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
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = ReduceLROnPlateau(
            opt, mode="min", factor=lr_factor, patience=lr_patience, verbose=verbose
        )

        early_stopping = EarlyStopping(
            patience=stop_patience, delta=int(stop_min_delta)
        )

        if self.dz:
            self.data_loader = class_mi_data_loader(
                self.X, self.Y, self.Z, val_size=val_size, device=self.device
            )
        else:
            self.data_loader = class_mi_data_loader(
                self.X, self.Y, val_size=val_size, device=self.device
            )

        self.loss_fn = nn.BCEWithLogitsLoss()
        val_loss_epoch = []
        train_loss_epoch = []
        train_acc_epoch = []
        val_acc_epoch = []
        val_dkl = []
        train_dkl = []

        for i in tqdm(range(max_epochs), disable=not verbose):
            # training
            self.train()
            for data, labels in self.data_loader.batch(batch_size=batch_size):
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    logits, _ = self.forward(data)
                    loss = self.loss_fn(logits, labels)
                    loss.backward()
                    opt.step()

            # validate and testing
            torch.set_grad_enabled(False)
            self.eval()
            with torch.no_grad():
                dkl, loss, acc = self.calc_mi_fn(
                    self.data_loader.train_data, self.data_loader.train_labels
                )
                train_dkl.append(dkl.item())
                train_loss_epoch.append(loss.item())
                train_acc_epoch.append(acc.item())
                dkl, loss, acc = self.calc_mi_fn(
                    self.data_loader.val_data, self.data_loader.val_labels
                )
                val_dkl.append(dkl.item())
                val_loss_epoch.append(loss.item())
                val_acc_epoch.append(acc.item())
                # learning rate scheduler
                scheduler.step(acc.item())
                # early stopping
                # early_stopping(-acc)
                early_stopping(loss)

            if early_stopping.early_stop:
                break

        self.train_dkl = np.array(train_dkl)
        self.train_loss_epoch = np.array(train_loss_epoch)
        self.train_acc_epoch = np.array(train_acc_epoch)
        self.val_dkl = np.array(val_dkl)
        self.val_loss_epoch = np.array(val_loss_epoch)
        self.val_acc_epoch = np.array(val_acc_epoch)

    def calc_mi_fn(self, samples, labels):
        logit, probs = self.forward(samples)
        # get loss function
        loss = self.loss_fn(logit, labels)
        # Calculate accuracy
        y_pred = torch.round(probs)
        acc = torch.sum(y_pred == labels) / labels.shape[0]

        labels = labels > 0
        likel_ratio_p = (probs[labels] + EPS) / (1 - probs[labels] - EPS)
        likel_ratio_q = (probs[torch.logical_not(labels)] + EPS) / (
            1 - probs[torch.logical_not(labels)] - EPS
        )
        fp = torch.log(torch.abs(likel_ratio_p))
        fq = torch.log(torch.abs(likel_ratio_q))

        Dkl = fp.mean() - (torch.logsumexp(fq, 0) - math.log(fq.shape[0]))
        return Dkl, loss, acc

    def get_mi(self):
        mi, _, _ = self.calc_mi_fn(self.data_loader.data, self.data_loader.labels)
        return mi.item()

    def get_curves(self):
        return (
            self.train_dkl,
            self.val_dkl,
            self.train_loss_epoch,
            self.val_loss_epoch,
            self.train_acc_epoch,
            self.val_acc_epoch,
        )
