#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification conditional mutual information
Generator approach

@inproceedings{
  title={CCMI: Classifier based conditional mutual information estimation},
  author={Mukherjee, Sudipto and Asnani, Himanshu and Kannan, Sreeram},
  booktitle={Uncertainty in artificial intelligence},
  pages={1083--1093},
  year={2020},
  organization={PMLR}
}
"""


import numpy as np
import torch
import torch.nn as nn

from minepy.class_mi.class_mi import ClassMiModel
from minepy.class_mi.class_mi_tools import class_cmi_gen_data_loader
from minepy.minepy_tools import toColVector

EPS = 1e-6


class ClassGenCMI(nn.Module):
    def __init__(self, X, Y, Z, hidden_layers=[64, 32], afn="gelu", device=None):
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
        # model
        self.model = ClassMiModel(
            input_dim=dx + dy + dz,
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
        self.data_loader = class_cmi_gen_data_loader(
            self.X, self.Y, self.Z, val_size=val_size, device=self.device
        )
        # Estimate I(X,Y|Z)
        (
            self.val_dkl_epoch,
            self.val_loss_epoch,
            self.val_loss_smoot_epoch,
        ) = self.model.fit_model(
            self.data_loader.train_samples,
            self.data_loader.train_labels,
            self.data_loader.val_samples,
            self.data_loader.val_labels,
            **fit_params
        )

    def get_cmi(self):
        return self.val_dkl_epoch.max()

    def get_curves(self):
        return (
            self.val_dkl_epoch,
            self.val_loss_epoch,
            self.val_loss_smoot_epoch,
        )
