#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import schedulefree
import torch
import torch.nn as nn
from scipy.special import betaln, digamma
from scipy.stats import beta
from tqdm import tqdm

from minepy.mi_nee.mi_nee_tools import hnee_data_loader
from minepy.minepy_tools import (
    EarlyStopping,
    ExpMovingAverageSmooth,
    get_activation_fn,
    toColVector,
)

EPS = 1e-10


class ReferenceDistribution:
    def __init__(
        self,
        X,
        sample_size,
        dist="uniform",
        device=None,
    ):
        """
        Parameters:
            dist: str, one of ["uniform", "gaussian", "uniform_clip"]
        """

        self.args: Optional[Tuple[Any, Any]] = None  # for type safety

        self.dist = dist
        self.device = device
        self.dim = X.shape[1]
        self.ref_sample_size = sample_size
        if self.dist == "uniform":
            self.args = (
                torch.tensor(X.min(axis=0), device=self.device),
                torch.tensor(X.max(axis=0), device=self.device),
            )
        elif self.dist == "uniform_clip":
            q1 = np.percentile(X, 1, axis=0)
            q99 = np.percentile(X, 99, axis=0)
            self.args = (
                torch.tensor(q1, device=self.device),
                torch.tensor(q99, device=self.device),
            )
        elif self.dist == "gaussian":
            self.args = (
                torch.tensor(X.mean(axis=0)).to(self.device),
                torch.tensor(X.std(axis=0)).to(self.device),
            )
        elif self.dist == "beta":
            # Scale to [0, 1] then fit shape parameters α and β
            from scipy.stats import beta

            X_np = np.asarray(X, dtype=np.float64)
            X_std = (X_np - X_np.min(0)) / (X_np.max(0) - X_np.min(0) + 1e-10)
            a, b, _, _ = beta.fit(X_std.flatten(), floc=0, fscale=1)
            self.args = (a, b)
        elif self.dist == "lognormal":
            X_clipped = np.clip(X, 1e-5, None)
            X_log = np.log(X_clipped)
            mu = np.mean(X_log, axis=0)
            sigma = np.std(X_log, axis=0)
            self.args = (
                torch.tensor(mu, device=self.device),
                torch.tensor(sigma, device=self.device),
            )
        else:
            raise ValueError(f"Unsupported reference distribution: {self.dist}")
        self.ref_sample = self.generate_samples()
        self.href = self.get_ref_entropy()

    def generate_samples(self) -> torch.Tensor:
        assert self.args is not None, "ReferenceDistribution args not initialized"
        if self.dist in ["uniform", "uniform_clip"]:
            lower, upper = self.args
            ref_samples = (
                torch.rand((self.ref_sample_size, self.dim), device=self.device)
                * (upper - lower)
                + lower
            )
        elif self.dist == "gaussian":
            mu, sigma = self.args
            ref_samples = (
                torch.randn((self.ref_sample_size, self.dim), device=self.device)
                * sigma
                + mu
            )
        elif self.dist == "beta":
            a, b = self.args
            samples = np.asarray(
                beta.rvs(a, b, size=(self.ref_sample_size, self.dim))
            ).astype(np.float32)
            ref_samples = torch.tensor(samples, device=self.device)
        elif self.dist == "lognormal":
            mu, sigma = self.args
            normal_samples = torch.randn(
                (self.ref_sample_size, self.dim), device=self.device
            )
            ref_samples = torch.exp(normal_samples * sigma + mu)
        else:
            raise ValueError(f"Unsupported reference distribution: {self.dist}")

        return ref_samples

    def sample(self, batch_size):
        indx = torch.randint(0, self.ref_sample_size, (batch_size,))
        return self.ref_sample[indx]

    def get_ref_entropy(self):
        assert self.args is not None, "ReferenceDistribution args not initialized"
        if self.dist == "uniform":
            lower, upper = self.args
            entropy = torch.sum(torch.log(upper - lower))
        elif self.dist == "gaussian":
            _, sigma = self.args
            entropy = 0.5 * torch.sum(torch.log(2 * math.pi * math.e * sigma**2))
        elif self.dist == "beta":
            a, b = self.args
            # Differential entropy of beta(a, b) on [0,1]
            entropy = (
                -torch.tensor(betaln(a, b))
                - (a - 1) * torch.tensor(digamma(a))
                - (b - 1) * torch.tensor(digamma(b))
                + (a + b - 2) * torch.tensor(digamma(a + b))
            ).sum()
        elif self.dist == "lognormal":
            mu, sigma = self.args
            entropy = torch.sum(mu + 0.5 * torch.log(2 * math.pi * math.e * sigma**2))
        else:
            raise ValueError(f"Unsupported reference distribution: {self.dist}")
        return entropy.to(self.device)


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
        reference_dist="uniform",
        reference_sample_size=2048,
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
        self.ref_dist = ReferenceDistribution(
            self.X,
            sample_size=reference_sample_size,
            dist=reference_dist,
            device=self.device,
        )

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
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
        # Smooth functions
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

        indicators = []
        for epoch in tqdm(range(max_epochs), disable=not verbose):
            # training
            rand_perm = torch.randperm(Xtrain.shape[0])
            self.train()
            opt.train()
            with torch.set_grad_enabled(True):
                for inds in rand_perm.split(batch_size, dim=0):
                    opt.zero_grad()
                    ref_samp = self.ref_dist.sample(batch_size)
                    train_loss = self.model(Xtrain[inds, :], ref_samp)
                    train_loss.backward()
                    opt.step()

            # validate and testing
            self.eval()
            opt.eval()
            with torch.set_grad_enabled(False):
                # validate
                ref_samp = self.ref_dist.sample(Xval.shape[0])
                val_loss = self.model(Xval, ref_samp)
                val_h = self.ref_dist.href + val_loss
                val_ema_loss = val_loss_ema_smooth(val_loss.item())
                val_ema_h = val_h_ema_smooth(val_h.item())

                # testing
                ref_samp = self.ref_dist.sample(X.shape[0])
                test_loss = self.model(X, ref_samp)
                test_h = self.ref_dist.href + test_loss
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
