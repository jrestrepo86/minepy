#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification conditional mutual information
generator approach
"""


import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from minepy.class_mi.class_mi_tools import (class_cmi_batch, get_ycondz,
                                            split_data_train_val)
from minepy.minepy_tools import EarlyStopping, get_activation_fn

EPS = 1e-6


class ClassCMI(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=50, num_hidden_layers=2, afn="elu", device=None
    ):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)
        # create model
        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(num_hidden_layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*seq)
        self.model = self.model.to(self.device)

    def forward(self, x, y, z, y_cz):
        n = x.shape[0]

        # samples from joint distribution
        s_joint = torch.cat((x, y, z), dim=1)
        # samples from product of marginal distribution
        s_cond = torch.cat((x, y_cz, z), dim=1)

        samples = torch.cat((s_joint, s_cond), dim=0)
        labels = torch.cat((torch.ones(n), torch.zeros(n)), dim=0).to(x.device)

        # random ordering
        inds = torch.randperm(2 * n)
        samples = samples[inds, :]
        labels = labels[inds]

        logit = torch.squeeze(self.model(samples))
        probs = torch.sigmoid(logit)

        return logit, labels, probs

    def fit(
        self,
        X,
        Y,
        Z,
        knn=1,
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
        self.knn = knn
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
        (
            Xtrain,
            Ytrain,
            Ztrain,
            Xval,
            Yval,
            Zval,
            self.Xtest,
            self.Ytest,
            self.Ztest,
        ) = split_data_train_val(X, Y, Z, val_size, self.device)

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
            for x, y, z, y_cz in class_cmi_batch(
                Xtrain, Ytrain, Ztrain, self.knn, batch_size=batch_size
            ):
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    logits, labels, _ = self.forward(x, y, z, y_cz)
                    loss = self.loss_fn(logits, labels)
                    loss.backward()
                    opt.step()

            # validate and testing
            torch.set_grad_enabled(False)
            self.eval()
            with torch.no_grad():
                dkl, loss, acc = self.calc_cmi_fn(Xtrain, Ytrain, Ztrain)
                train_dkl.append(dkl.item())
                train_loss_epoch.append(loss.item())
                train_acc_epoch.append(acc.item())
                dkl, loss, acc = self.calc_cmi_fn(Xval, Yval, Zval)
                val_dkl.append(dkl.item())
                val_loss_epoch.append(loss.item())
                val_acc_epoch.append(acc.item())
                # learning rate scheduler
                # scheduler.step(acc.item())
                # early stopping
                # early_stopping(-acc)

            # if early_stopping.early_stop:
            #     break

        self.train_dkl = np.array(train_dkl)
        self.train_loss_epoch = np.array(train_loss_epoch)
        self.train_acc_epoch = np.array(train_acc_epoch)
        self.val_dkl = np.array(val_dkl)
        self.val_loss_epoch = np.array(val_loss_epoch)
        self.val_acc_epoch = np.array(val_acc_epoch)

    def calc_cmi_fn(self, x, y, z):
        x, y, z, y_cz = get_ycondz(x, y, z, k=1)

        logit, labels, probs = self.forward(x, y, z, y_cz)
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
        cmi, _, _ = self.calc_cmi_fn(self.Xtest, self.Ytest, self.Ztest)
        return cmi

    def get_curves(self):
        return (
            self.train_dkl,
            self.val_dkl,
            self.train_loss_epoch,
            self.val_loss_epoch,
            self.train_acc_epoch,
            self.val_acc_epoch,
        )
