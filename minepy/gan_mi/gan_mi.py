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
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from minepy.minepy_tools import (EarlyStopping, ExpMovingAverageSmooth,
                                 get_activation_fn, toColVector)

EPS = 1e-6


class CGanModel(nn.Module):
    def __init__(
        self,
        g_input_dim,
        g_output_dim,
        r_input_dim,
        g_hidden_layers=[64, 32],
        g_afn="gelu",
        r_hidden_layers=[32, 16],
        r_afn="gelu",
        device=None,
    ):
        super().__init__()

        g_hidden_layers = [int(hl) for hl in g_hidden_layers]
        r_hidden_layers = [int(hl) for hl in r_hidden_layers]

        # generator
        activation_fn = get_activation_fn(g_afn)
        seq = [nn.Linear(g_input_dim, g_hidden_layers[0]), activation_fn()]
        for i in range(len(g_hidden_layers) - 1):
            seq += [
                nn.Linear(g_hidden_layers[i], g_hidden_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(g_hidden_layers[-1], g_output_dim)]
        self.generator = nn.Sequential(*seq).to(device)

        # regresor
        activation_fn = get_activation_fn(r_afn)
        seq = [nn.Linear(r_input_dim, r_hidden_layers[0]), activation_fn()]
        for i in range(len(r_hidden_layers) - 1):
            seq += [
                nn.Linear(r_hidden_layers[i], r_hidden_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(r_hidden_layers[-1], 1)]
        self.regresor = nn.Sequential(*seq).to(device)

    def generator_forward(self, samples):
        return self.generator(samples)

    def regresor_forward(self, samples):
        return self.regresor(samples)


class GanMI(nn.Module):
    def __init__(
        self,
        X,
        Y,
        noise_dim=40,
        g_hidden_layers=[64, 32, 16, 8],
        g_afn="gelu",
        r_hidden_layers=[32, 16, 8],
        r_afn="gelu",
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
        self.X = torch.from_numpy(toColVector(X.astype(np.float32))).to(self.device)
        self.Y = torch.from_numpy(toColVector(Y.astype(np.float32))).to(self.device)
        dx = self.X.shape[1]
        dy = self.Y.shape[1]
        self.noise_dim = noise_dim
        # cmigan model
        self.model = CGanModel(
            g_input_dim=noise_dim,
            g_output_dim=dy,
            g_hidden_layers=g_hidden_layers,
            g_afn=g_afn,
            r_input_dim=dx + dy,
            r_hidden_layers=r_hidden_layers,
            r_afn=r_afn,
            device=self.device,
        )

    def generator_loss(self, r_marginal):
        return -(torch.logsumexp(r_marginal, 0) - math.log(r_marginal.shape[0]))

    def regresor_loss(self, r_joint, r_marginal):
        joint_loss = r_joint.mean()
        marg_loss = torch.logsumexp(r_marginal, 0) - math.log(r_marginal.shape[0])
        return -joint_loss + marg_loss

    def fit(
        self,
        batch_size=64,
        max_epochs=40000,
        lr=1e-6,
        weight_decay=5e-5,
        stop_patience=1000,
        stop_min_delta=0.0,
        r_training_steps=5,
        g_training_steps=1,
        verbose=False,
    ):
        self.stop_patience = stop_patience

        gen_opt = torch.optim.RMSprop(
            self.model.generator.parameters(), lr=lr, weight_decay=weight_decay
        )
        reg_opt = torch.optim.RMSprop(
            self.model.regresor.parameters(), lr=lr, weight_decay=weight_decay
        )

        gen_scheduler = CyclicLR(gen_opt, base_lr=lr, max_lr=1e-3, mode="triangular2")
        reg_scheduler = CyclicLR(reg_opt, base_lr=lr, max_lr=1e-3, mode="triangular2")

        early_stopping = EarlyStopping(
            patience=stop_patience, delta=int(stop_min_delta)
        )

        reg_loss_ema_smooth = ExpMovingAverageSmooth()

        mi_epoch = []
        gen_loss_epoch = []
        reg_loss_epoch = []
        reg_loss_smooth_epoch = []

        # training
        self.train()
        for _ in tqdm(range(max_epochs), disable=not verbose):
            n = self.X.shape[0]
            # train regresor
            for _ in range(r_training_steps):
                inds = torch.randperm(n)
                # torch.randint(0, n, (batch_size,))
                X = self.X[inds, :]
                Y = self.Y[inds, :]
                with torch.set_grad_enabled(True):
                    reg_opt.zero_grad()
                    noise = torch.normal(0, 1, size=(n, self.noise_dim)).to(self.device)
                    Ygen = self.model.generator_forward(noise)
                    joint_samples = torch.cat((X, Y), dim=1)
                    marg_samples = torch.cat((X, Ygen), dim=1)
                    reg_join_out = self.model.regresor_forward(joint_samples)
                    reg_marg_out = self.model.regresor_forward(marg_samples)
                    reg_loss = self.regresor_loss(reg_join_out, reg_marg_out)
                    reg_loss.backward()
                    reg_opt.step()
            reg_loss_epoch.append(reg_loss.item())
            # train generator
            for _ in range(g_training_steps):
                with torch.set_grad_enabled(True):
                    gen_opt.zero_grad()
                    noise = torch.normal(0, 1, size=(n, self.noise_dim)).to(self.device)
                    Ygen = self.model.generator_forward(noise)
                    marg_samples = torch.cat((X, Ygen), dim=1)
                    reg_marg_out = self.model.regresor_forward(marg_samples)
                    gen_loss = self.generator_loss(reg_marg_out)
                    gen_loss.backward()
                    gen_opt.step()
            gen_loss_epoch.append(gen_loss.item())
            # validate and testing
            self.eval()
            with torch.set_grad_enabled(False):
                noise = torch.normal(0, 1, size=(n, self.noise_dim)).to(self.device)
                Ygen = self.model.generator_forward(noise)
                joint_samples = torch.cat((self.X, self.Y), dim=1)
                marg_samples = torch.cat((self.X, Ygen), dim=1)
                reg_join_out = self.model.regresor_forward(joint_samples)
                reg_marg_out = self.model.regresor_forward(marg_samples)
                reg_loss = self.regresor_loss(reg_join_out, reg_marg_out)
                # learning rate scheduler
                gen_scheduler.step()
                reg_scheduler.step()
                # smooth reg loss
                reg_loss_smooth = reg_loss_ema_smooth(reg_loss)
                reg_loss_smooth_epoch.append(reg_loss_smooth.item())
                # early_stopping
                early_stopping(reg_loss_smooth)

            mi_epoch.append(-reg_loss.item())
            gen_scheduler.step()
            reg_scheduler.step()
            if early_stopping.early_stop:
                break

        self.mi_epoch = np.array(mi_epoch)
        self.gen_loss_epoch = np.array(gen_loss_epoch)
        self.reg_loss_epoch = np.array(reg_loss_epoch)
        self.reg_loss_smooth_epoch = np.array(reg_loss_smooth_epoch)

    def get_mi(self):
        return -self.reg_loss_smooth_epoch[-1]

    def get_curves(self):
        return (
            self.mi_epoch,
            self.gen_loss_epoch,
            self.reg_loss_epoch,
            self.reg_loss_smooth_epoch,
        )
