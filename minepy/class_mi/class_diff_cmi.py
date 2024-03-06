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

from minepy.class_mi.class_mi import ClassMiModel
from minepy.class_mi.class_mi_tools import class_cmi_diff_data_loader
from minepy.minepy_tools import toColVector

EPS = 1e-6


class ClassDiffCMI(nn.Module):
    def __init__(
        self,
        X,
        Y,
        Z,
        hidden_layers_xyz=[64, 32],
        hidden_layers_xz=[32, 16],
        afn="gelu",
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
        self.Z = toColVector(Z.astype(np.float32))
        dx = self.X.shape[1]
        dy = self.Y.shape[1]
        dz = self.Z.shape[1]
        # model for I(X,Y,Z)
        self.model_xyz = ClassMiModel(
            input_dim=dx + dy + dz,
            hidden_layers=hidden_layers_xyz,
            afn=afn,
            device=self.device,
        )
        # model for I(X,Z)
        self.model_xz = ClassMiModel(
            input_dim=dx + dz,
            hidden_layers=hidden_layers_xz,
            afn=afn,
            device=self.device,
        )

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-4,
        weight_decay=5e-5,
        stop_patience=1000,
        stop_min_delta=0.0,
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

        # Data loader
        self.data_loader = class_cmi_diff_data_loader(
            self.X, self.Y, self.Z, val_size=val_size, device=self.device
        )

        # Estimate I(X,Y,Z)
        (
            self.val_dkl_epoch_xyz,
            self.val_loss_epoch_xyz,
            _,
        ) = self.model_xyz.fit_model(
            self.data_loader.train_samples_xyz,
            self.data_loader.train_labels_xyz,
            self.data_loader.val_samples_xyz,
            self.data_loader.val_labels_xyz,
            **fit_params,
        )

        # Estimate I(X,Z)
        (
            self.val_dkl_epoch_xz,
            self.val_loss_epoch_xz,
            _,
        ) = self.model_xz.fit_model(
            self.data_loader.train_samples_xz,
            self.data_loader.train_labels_xz,
            self.data_loader.val_samples_xz,
            self.data_loader.val_labels_xz,
            **fit_params,
        )

    def get_cmi(self):
        return self.val_dkl_epoch_xyz.max() - self.val_dkl_epoch_xz.max()

    def get_curves(self):
        min_epoch = np.minimum(self.val_dkl_epoch_xyz.size, self.val_dkl_epoch_xz.size)
        return (
            self.val_dkl_epoch_xyz[:min_epoch] - self.val_dkl_epoch_xz[:min_epoch],
            self.val_dkl_epoch_xyz[:min_epoch],
            self.val_loss_epoch_xyz[:min_epoch],
            self.val_dkl_epoch_xz[:min_epoch],
            self.val_loss_epoch_xz[:min_epoch],
        )
