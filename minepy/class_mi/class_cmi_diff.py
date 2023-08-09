#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification conditional mutual information
mi difference approach
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from minepy.class_mi.class_mi import ClassMI
from minepy.minepy_tools import EarlyStopping, toColVector

EPS = 1e-6


class ClassCMIDiff(nn.Module):
    def __init__(
        self, X, Y, Z, hidden_dim=50, num_hidden_layers=2, afn="elu", device=None
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
        self.Z = toColVector(Z.astype(np.float32))
        # setup models
        self.mi_xyz = ClassMI(
            self.X,
            self.Y,
            self.Z,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            afn=afn,
            device=device,
        )
        self.mi_xz = ClassMI(
            self.X,
            self.Z,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            afn=afn,
            device=device,
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
            "val_size": val_size,
            "lr": lr,
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "stop_patience": stop_patience,
            "stop_min_delta": stop_min_delta,
            "weight_decay": weight_decay,
            "verbose": verbose,
        }
        # train I(X,Y,Z) and I(X,Z)
        self.mi_xyz.fit(**fit_params)
        self.mi_xz.fit(**fit_params)
        (
            self.Dkl_train_xyz,
            self.Dkl_val_xyz,
            self.train_loss_xyz,
            self.val_loss_xyz,
            self.train_acc_xyz,
            self.val_acc_xyz,
        ) = self.mi_xyz.get_curves()
        (
            self.Dkl_train_xz,
            self.Dkl_val_xz,
            self.train_loss_xz,
            self.val_loss_xz,
            self.train_acc_xz,
            self.val_acc_xz,
        ) = self.mi_xz.get_curves()

    def get_cmi(self):
        mi_xyz = self.mi_xyz.get_mi()
        data = self.mi_xyz.data_loader.data_p
        labels = self.mi_xyz.data_loader.labels_p
        mi_xz = self.mi_xz.get_mi(data=data, labels=labels)
        return mi_xyz - mi_xz

    def get_curves(self):
        min_epoch = np.minimum(self.Dkl_train_xyz.size, self.Dkl_train_xz.size)
        cmi_train = self.Dkl_train_xyz[:min_epoch] - self.Dkl_train_xz[:min_epoch]
        min_epoch = np.minimum(self.Dkl_val_xyz.size, self.Dkl_val_xz.size)
        cmi_val = self.Dkl_val_xyz[:min_epoch] - self.Dkl_val_xz[:min_epoch]
        return (
            cmi_train,
            cmi_val,
            self.Dkl_train_xyz,
            self.Dkl_val_xyz,
            self.train_loss_xyz,
            self.val_loss_xyz,
            self.train_acc_xyz,
            self.val_acc_xyz,
            self.Dkl_train_xz,
            self.Dkl_val_xz,
            self.train_loss_xz,
            self.val_loss_xz,
            self.train_acc_xz,
            self.val_acc_xz,
        )
