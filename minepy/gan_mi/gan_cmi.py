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
from torch.optim.lr_scheduler import StepLR
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
        g_afn="relu",
        r_hidden_layers=[32, 16],
        r_afn="relu",
        device=None,
    ):
        super().__init__()

        g_hidden_layers = [int(x) for x in g_hidden_layers]
        r_hidden_layers = [int(x) for x in r_hidden_layers]

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


class ClassCMiGan(nn.Module):
    def __init__(
        self,
        X,
        Y,
        Z,
        noise_dim=40,
        g_hidden_layers=[64, 32],
        g_afn="relu",
        r_hidden_layers=[32, 16],
        r_afn="relu",
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
        self.Z = torch.from_numpy(toColVector(Z.astype(np.float32))).to(self.device)
        dx = self.X.shape[1]
        dy = self.Y.shape[1]
        dz = self.Z.shape[1]
        self.noise_dim = noise_dim
        # cmigan model
        self.model = CGanModel(
            g_input_dim=noise_dim + dz,
            g_output_dim=dy,
            g_hidden_layers=g_hidden_layers,
            g_afn=g_afn,
            r_input_dim=dx + dy + dz,
            r_hidden_layers=r_hidden_layers,
            r_afn=r_afn,
            device=self.device,
        )

    def generator_loss(self, r_marginal):
        return -(torch.logsumexp(r_marginal, 0) - math.log(r_marginal.shape[0]))

    def regresor_loss(self, r_joint, r_marginal):
        joint_loss = r_joint.mean()
        marg_loss = torch.logsumexp(r_marginal, 0) - math.log(r_marginal.shape[0])
        self.ref_marg_loss = marg_loss
        return -joint_loss + marg_loss

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        r_training_steps=3,
        g_training_steps=1,
        val_size=0.2,
        lr=1e-4,
        lr_factor=0.1,
        lr_patience=10,
        stop_patience=100,
        stop_min_delta=0.05,
        weight_decay=5e-5,
        verbose=False,
    ):
        gen_opt = torch.optim.RMSprop(
            self.model.generator.parameters(), lr=lr, weight_decay=weight_decay
        )
        reg_opt = torch.optim.RMSprop(
            self.model.regresor.parameters(), lr=lr, weight_decay=weight_decay
        )

        # reg_scheduler = StepLR(reg_opt, step_size=lr_patience, gamma=lr_factor)
        # gen_scheduler = StepLR(gen_opt, step_size=lr_patience, gamma=lr_factor)

        cmi_epoch = []
        gen_loss_epoch = []
        reg_loss_epoch = []
        reg_marg_loss_epoch = []
        # training
        self.train()
        for _ in tqdm(range(max_epochs), disable=not verbose):
            n = self.X.shape[0]
            # train regresor
            for _ in range(r_training_steps):
                inds = torch.randint(0, n, (batch_size,))
                X = self.X[inds, :]
                Y = self.Y[inds, :]
                Z = self.Z[inds, :]
                with torch.set_grad_enabled(True):
                    reg_opt.zero_grad()
                    noise = torch.normal(0, 1, size=(batch_size, self.noise_dim)).to(
                        self.device
                    )
                    Ygen = self.model.generator_forward(torch.cat((noise, Z), dim=1))
                    joint_samples = torch.cat((X, Y, Z), dim=1)
                    marg_samples = torch.cat((X, Ygen, Z), dim=1)
                    reg_join_out = self.model.regresor_forward(joint_samples)
                    reg_marg_out = self.model.regresor_forward(marg_samples)
                    reg_loss = self.regresor_loss(reg_join_out, reg_marg_out)
                    reg_loss.backward()
                    reg_opt.step()
            reg_loss_epoch.append(reg_loss.item())
            reg_marg_loss_epoch.append(self.ref_marg_loss.item())
            # train generator
            for _ in range(g_training_steps):
                # inds = torch.randint(0, n, (batch_size,))
                # X = self.X[inds, :]
                # Y = self.Y[inds, :]
                # Z = self.Z[inds, :]
                with torch.set_grad_enabled(True):
                    gen_opt.zero_grad()
                    noise = torch.normal(0, 1, size=(batch_size, self.noise_dim)).to(
                        self.device
                    )
                    Ygen = self.model.generator_forward(torch.cat((noise, Z), dim=1))
                    marg_samples = torch.cat((X, Ygen, Z), dim=1)
                    reg_marg_out = self.model.regresor_forward(marg_samples)
                    gen_loss = self.generator_loss(reg_marg_out)
                    gen_loss.backward()
                    gen_opt.step()
            gen_loss_epoch.append(gen_loss.item())
            # validate and testing
            self.eval()
            with torch.set_grad_enabled(False):
                noise = torch.normal(0, 1, size=(n, self.noise_dim)).to(self.device)
                Ygen = self.model.generator_forward(torch.cat((noise, self.Z), dim=1))
                joint_samples = torch.cat((self.X, self.Y, self.Z), dim=1)
                marg_samples = torch.cat((self.X, Ygen, self.Z), dim=1)
                reg_join_out = self.model.regresor_forward(joint_samples)
                reg_marg_out = self.model.regresor_forward(marg_samples)
                reg_loss = self.regresor_loss(reg_join_out, reg_marg_out)
                cmi_epoch.append(-reg_loss.item())
        self.cmi_epoch = np.array(cmi_epoch)
        self.gen_loss_epoch = np.array(gen_loss_epoch)
        self.reg_loss_epoch = np.array(reg_loss_epoch)
        self.reg_marg_loss_epoch = np.array(reg_marg_loss_epoch)

    def get_cmi(self):
        return self.cmi_epoch[-1000:].mean()

    def get_curves(self):
        return (
            self.cmi_epoch,
            self.gen_loss_epoch,
            self.reg_loss_epoch,
            self.reg_marg_loss_epoch,
        )
