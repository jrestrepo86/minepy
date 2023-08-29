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

from minepy.class_mi.class_mi import ClassMiModel
from minepy.class_mi.class_mi_tools import class_cmi_diff_data_loader
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
        dx = self.X.shape[1]
        dy = self.Y.shape[1]
        dz = self.Z.shape[1]
        # model for I(X,Y,Z)
        self.model_xyz = ClassMiModel(
            input_dim=dx + dy + dz,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            afn=afn,
            device=self.device,
        )
        # model for I(X,Z)
        self.model_xz = ClassMiModel(
            input_dim=dx + dz,
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

        self.data_loader = class_cmi_diff_data_loader(
            self.X, self.Y, self.Z, val_size=val_size, device=self.device
        )
        # I(X,Y,Z)
        (
            self.val_dkl_xyz,
            self.val_loss_epoch_xyz,
            _,
        ) = self.model_xyz.fit_model(
            self.data_loader.train_samples_xyz,
            self.data_loader.train_labels_xyz,
            self.data_loader.val_samples_xyz,
            self.data_loader.val_labels_xyz,
            **fit_params
        )
        # I(X,Z)
        (
            self.val_dkl_xz,
            self.val_loss_epoch_xz,
            _,
        ) = self.model_xz.fit_model(
            self.data_loader.train_samples_xz,
            self.data_loader.train_labels_xz,
            self.data_loader.val_samples_xz,
            self.data_loader.val_labels_xz,
            **fit_params
        )

    def get_cmi(self):
        mi_xyz_test, _ = self.model_xyz.calc_mi_fn(
            self.data_loader.samples_xyz, self.data_loader.labels_xyz
        )
        mi_xz_test, _ = self.model_xz.calc_mi_fn(
            self.data_loader.samples_xz, self.data_loader.labels_xz
        )
        mi_xyz_val, _ = self.model_xyz.calc_mi_fn(
            self.data_loader.val_samples_xyz, self.data_loader.val_labels_xyz
        )
        mi_xz_val, _ = self.model_xz.calc_mi_fn(
            self.data_loader.val_samples_xz, self.data_loader.val_labels_xz
        )
        cmi_test = mi_xyz_test - mi_xz_test
        cmi_val = mi_xyz_val - mi_xz_val
        return cmi_test, cmi_val

    def get_curves(self):
        min_epoch = np.minimum(self.val_dkl_xyz.size, self.val_dkl_xz.size)
        cmi_val = self.val_dkl_xyz[:min_epoch] - self.val_dkl_xz[:min_epoch]
        return (
            cmi_val,
            self.val_dkl_xyz,
            self.val_loss_epoch_xyz,
            self.val_dkl_xz,
            self.val_loss_epoch_xz,
        )
