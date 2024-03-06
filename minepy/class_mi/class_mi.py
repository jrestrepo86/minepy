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
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from minepy.class_mi.class_mi_tools import class_mi_data_loader
from minepy.minepy_tools import (
    EarlyStopping,
    ExpMovingAverageSmooth,
    get_activation_fn,
    toColVector,
)

EPS = 1e-10


class ClassMiModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32], afn="gelu", device=None):
        super().__init__()

        hidden_layers = [int(x) for x in hidden_layers]

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_layers[0]), activation_fn()]
        for i in range(len(hidden_layers) - 1):
            seq += [
                nn.Dropout(0.2),
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(hidden_layers[-1], 2)]
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
        weight_decay=5e-5,
        stop_patience=100,
        stop_min_delta=0,
        verbose=False,
    ):
        opt = torch.optim.RMSprop(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = CyclicLR(opt, base_lr=lr, max_lr=1e-3, mode="triangular2")
        early_stopping = EarlyStopping(
            patience=int(stop_patience), delta=stop_min_delta
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

            # validation
            self.eval()
            with torch.set_grad_enabled(False):
                # validation set
                val_dkl, val_loss = self.calc_mi_fn(val_samples, val_labels)
                val_dkl_epoch.append(val_dkl.item())
                val_loss_epoch.append(val_loss.item())
                val_loss_smooth = val_loss_ema_smooth(val_loss)
                val_loss_smooth_epoch.append(val_loss_smooth.item())

                # learning rate scheduler
                scheduler.step()
                # early stopping
                # early_stopping(val_loss_smooth)

            if early_stopping.early_stop:
                break

        return (
            np.array(val_dkl_epoch),
            np.array(val_loss_epoch),
            np.array(val_loss_smooth_epoch),
        )


class ClassMI(nn.Module):
    def __init__(self, X, Y, hidden_layers=[64, 32], afn="elu", device=None):
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
            hidden_layers=hidden_layers,
            afn=afn,
            device=self.device,
        )

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-4,
        weight_decay=5e-5,
        stop_patience=100,
        stop_min_delta=0.05,
        val_size=0.2,
        verbose=False,
    ):
        fit_params = {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "stop_patience": stop_patience,
            "stop_min_delta": stop_min_delta,
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
            **fit_params,
        )

    def get_mi(self):
        return np.mean(self.val_dkl_epoch[-2000:])

    def get_curves(self):
        return (
            self.val_dkl_epoch,
            self.val_loss_epoch,
        )
