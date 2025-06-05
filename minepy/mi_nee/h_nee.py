#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math

import numpy as np
import pandas as pd
import plotly.express as px
import schedulefree
import torch
import torch.nn as nn
from tqdm import tqdm

from minepy.mi_nee.mi_nee_tools import hnee_data_loader
from minepy.minepy_tools import (
    EarlyStopping,
    ExpMovingAverageSmooth,
    get_activation_fn,
    toColVector,
)

EPS = 1e-10


class HNeeModel(nn.Module):
    """docstring for ."""

    def __init__(
        self,
        input_dim,
        hidden_layers,
        afn,
    ):
        super().__init__()

        hidden_layers = [int(hl) for hl in hidden_layers]

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_layers[0]), activation_fn()]
        for i in range(len(hidden_layers) - 1):
            seq += [
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(hidden_layers[-1], 1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x, xref):
        mean_f = self.net(x).mean()
        log_mean_ef_ref = torch.logsumexp(self.net(xref), 0) - math.log(xref.shape[0])
        div = mean_f - log_mean_ef_ref
        return -div


class HNee(nn.Module):
    def __init__(
        self,
        X,
        hidden_layers=[150, 150, 150],
        afn="elu",
        device=None,
    ):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)
        # Vars
        self.X = toColVector(X.astype(np.float32))
        self.dx = self.X.shape[1]
        self.model = HNeeModel(
            input_dim=self.dx,
            hidden_layers=hidden_layers,
            afn=afn,
        )
        self.model = self.model.to(self.device)

        self.ref_min = torch.tensor(X.min()).to(self.device)
        self.ref_max = torch.tensor(X.max()).to(self.device)
        self.href = torch.tensor(math.log(self.ref_max - self.ref_min)).to(self.device)

    def ref_sample_(self, x):
        ref_samp = torch.rand((self.n_ref_samples, x.shape[1]), device=self.device)
        return (self.ref_max - self.ref_min) * ref_samp + self.ref_min

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        ref_batch_factor=4,
        lr=1e-3,
        weight_decay=5e-5,
        warmup_epochs=100,
        stop_patience=500,
        stop_min_delta=0,
        val_size=0.2,
        random_sample=True,
        verbose=False,
    ):
        # Optimizer
        opt = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_epochs,
        )
        # Early stopping
        early_stopping = EarlyStopping(
            patience=stop_patience + warmup_epochs, delta=stop_min_delta
        )
        # Smooth val loss
        val_loss_ema_smooth = ExpMovingAverageSmooth()
        val_h_ema_smooth = ExpMovingAverageSmooth()
        test_loss_ema_smooth = ExpMovingAverageSmooth()
        test_h_ema_smooth = ExpMovingAverageSmooth()

        # Data
        Xtrain, Xval, X = hnee_data_loader(
            self.X, val_size=val_size, random_sample=random_sample, device=self.device
        )

        if batch_size == "full":
            batch_size = Xtrain.shape[0]
        self.n_ref_samples = int(batch_size * ref_batch_factor)

        indicators = []
        for epoch in tqdm(range(max_epochs), disable=not verbose):
            # training
            rand_perm = torch.randperm(Xtrain.shape[0])
            self.train()
            opt.train()
            for inds in rand_perm.split(batch_size, dim=0):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    ref_samp = self.ref_sample_(Xtrain[inds, :])
                    train_loss = self.model(Xtrain[inds, :], ref_samp)
                    train_loss.backward()
                    opt.step()

            # validate and testing
            self.eval()
            opt.eval()
            with torch.set_grad_enabled(False):
                # validate
                ref_samp = self.ref_sample_(Xval)
                val_loss = self.model(Xval, ref_samp)
                val_h = self.href + val_loss
                val_ema_loss = val_loss_ema_smooth(val_loss.item())
                val_ema_h = val_h_ema_smooth(val_h.item())

                # testing
                ref_samp = self.ref_sample_(X)
                test_loss = self.model(X, ref_samp)
                test_h = self.href + test_loss
                test_ema_loss = test_loss_ema_smooth(test_loss.item())
                test_ema_h = test_h_ema_smooth(test_h.item())

                indicators.append(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss.item(),
                        "val_ema_loss": val_ema_loss,
                        "val_h": val_h.item(),
                        "val_ema_h": val_ema_h,
                        "test_loss": test_loss.item(),
                        "test_ema_loss": test_ema_loss,
                        "test_h": test_h.item(),
                        "test_ema_h": test_ema_h,
                    }
                )

                early_stopping(val_ema_loss)  # early stopping

            if early_stopping.early_stop:
                break

        self.indicators = pd.DataFrame(indicators)

    def get_h(self, all=False):
        if not hasattr(self, "indicators") or self.indicators.empty:
            raise ValueError("No indicators available. Have you called .fit()?")

        min_ind = self.indicators["val_ema_loss"].idxmin()
        h_ema_test = self.indicators.at[min_ind, "test_ema_h"]
        h_test = self.indicators.at[min_ind, "test_h"]

        if all:
            h_val = self.indicators.at[min_ind, "val_h"]
            fepoch = self.indicators.at[min_ind, "epoch"]
            return h_val, h_test, h_ema_test, min_ind, fepoch
        else:
            return h_test

    def get_curves(self):
        if not hasattr(self, "indicators") or self.indicators.empty:
            raise ValueError("No indicators available. Have you called .fit()?")
        return self.indicators

    def plot_indicators(self):
        if not hasattr(self, "indicators") or self.indicators.empty:
            raise ValueError("No indicators to plot. Did you call .fit()?")

        # Reshape the DataFrame to long format for px.line
        df_plot = self.indicators.melt(
            id_vars="epoch",
            value_vars=[
                "val_loss",
                "val_ema_loss",
                "val_h",
                "val_ema_h",
                "test_loss",
                "test_ema_loss",
                "test_h",
                "test_ema_h",
            ],
            var_name="Indicator",
            value_name="Value",
        )

        fig = px.line(
            df_plot,
            x="epoch",
            y="Value",
            color="Indicator",
            markers=True,
            title="Training Indicators vs Epochs",
            labels={"epoch": "Epoch", "Value": "Value"},
        )

        fig.update_layout(template="plotly_white")
        fig.show()
