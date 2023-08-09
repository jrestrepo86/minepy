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
from minepy.minepy_tools import EarlyStopping, get_activation_fn, toColVector

EPS = 1e-6


class ClassMiModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=50, num_hidden_layers=2, afn="elu", device=None
    ):
        super().__init__()

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(num_hidden_layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*seq).to(device)

    def forward(self, samples):
        logit = torch.squeeze(self.net(samples))
        probs = torch.sigmoid(logit)

        return logit, probs

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
        stop_min_delta=0.05,
        weight_decay=5e-5,
        verbose=False,
    ):
        opt = torch.optim.Adam(
            self.net.parameters(),
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

        self.loss_fn = nn.BCEWithLogitsLoss()
        val_loss_epoch = []
        val_acc_epoch = []
        val_dkl = []

        for i in tqdm(range(max_epochs), disable=not verbose):
            # training
            self.train()
            for samples, labels in batch(
                train_samples, train_labels, batch_size=batch_size
            ):
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    logits, _ = self.forward(samples)
                    loss = self.loss_fn(logits, labels)
                    loss.backward()
                    opt.step()

            # validate and testing
            torch.set_grad_enabled(False)
            self.eval()
            with torch.no_grad():
                dkl, loss, acc = self.calc_mi_fn(val_samples, val_labels)
                val_dkl.append(dkl.item())
                val_loss_epoch.append(loss.item())
                val_acc_epoch.append(acc.item())
                # learning rate scheduler
                scheduler.step(acc.item())
                # early stopping
                early_stopping(loss)

            if early_stopping.early_stop:
                break

        return (np.array(val_dkl), np.array(val_loss_epoch), np.array(val_acc_epoch))


class ClassMI(nn.Module):
    def __init__(
        self,
        X,
        Y,
        Z=None,
        hidden_dim=50,
        num_hidden_layers=2,
        afn="elu",
        device=None,
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

        if self.dz:
            self.data_loader = class_mi_data_loader(
                self.X, self.Y, self.Z, val_size=val_size, device=self.device
            )
        else:
            self.data_loader = class_mi_data_loader(
                self.X, self.Y, val_size=val_size, device=self.device
            )

        self.val_dkl, self.val_loss_epoch, self.val_acc_epoch = self.model.fit_model(
            self.data_loader.train["samples"],
            self.data_loader.train["labels"],
            self.data_loader.val["samples"],
            self.data_loader.val["labels"],
            **fit_params
        )

    def get_mi(self, data=None, labels=None):
        if (data is None) or (labels is None):
            mi, _, _ = self.model.calc_mi_fn(
                self.data_loader.samples, self.data_loader.labels
            )
        else:
            mi, _, _ = self.model.calc_mi_fn(data, labels)
        return mi.item()

    def get_curves(self):
        return (
            self.val_dkl,
            self.val_loss_epoch,
            self.val_acc_epoch,
        )
