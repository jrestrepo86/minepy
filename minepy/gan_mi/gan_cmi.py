#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification conditional mutual information
generator approach

How to Train a GAN? Tips and tricks to make GANs work
https://github.com/soumith/ganhacks
"""


import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from minepy.gan_mi.gan_mi import CGanModel
from minepy.minepy_tools import (EarlyStopping, ExpMovingAverageSmooth,
                                 toColVector)


class GanCMI(nn.Module):
    def __init__(
        self,
        X,
        Y,
        Z,
        noise_dim=40,
        g_hidden_layers=[64, 32],
        g_afn="gelu",
        r_hidden_layers=[32, 16],
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
        return -joint_loss + marg_loss

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        g_base_lr=1e-8,
        g_max_lr=1e-5,
        r_base_lr=1e-5,
        r_max_lr=1e-3,
        weight_decay=5e-5,
        stop_patience=1000,
        stop_min_delta=0.0,
        r_training_steps=5,
        g_training_steps=1,
        verbose=False,
    ):
        gen_opt = torch.optim.RMSprop(
            self.model.generator.parameters(), lr=g_base_lr, weight_decay=weight_decay
        )

        reg_opt = torch.optim.RMSprop(
            self.model.regresor.parameters(), lr=r_base_lr, weight_decay=weight_decay
        )

        gen_scheduler = CyclicLR(
            gen_opt, base_lr=g_base_lr, max_lr=g_max_lr, mode="triangular2"
        )
        reg_scheduler = CyclicLR(
            reg_opt, base_lr=r_base_lr, max_lr=r_max_lr, mode="triangular2"
        )

        # early_stopping = EarlyStopping(
        #     patience=stop_patience, delta=int(stop_min_delta)
        # )

        reg_loss_ema_smooth = ExpMovingAverageSmooth()

        cmi_epoch = []
        gen_loss_epoch = []
        reg_loss_epoch = []
        reg_loss_smooth_epoch = []

        # training
        self.train()
        for i in tqdm(range(max_epochs), disable=not verbose):
            n = self.X.shape[0]
            if batch_size == "full":
                batch_size = n
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
            # train generator
            for _ in range(g_training_steps):
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
                # smooth reg loss
                reg_loss_smooth = reg_loss_ema_smooth(reg_loss)
                reg_loss_smooth_epoch.append(reg_loss_smooth.item())
                # early_stopping
                # if i >= 10000:  # skip generator initial drifting
                #     early_stopping(reg_loss_smooth)

            cmi_epoch.append(-reg_loss.item())
            gen_scheduler.step()
            reg_scheduler.step()
            # if early_stopping.early_stop:
            #     break

        self.cmi_epoch = np.array(cmi_epoch)
        self.gen_loss_epoch = np.array(gen_loss_epoch)
        self.reg_loss_epoch = np.array(reg_loss_epoch)
        self.reg_loss_smooth_epoch = np.array(reg_loss_smooth_epoch)

    def get_cmi(self):
        return -np.mean(self.reg_loss_smooth_epoch[-2000:])

    def get_curves(self):
        return (
            self.cmi_epoch,
            self.gen_loss_epoch,
            self.reg_loss_epoch,
            self.reg_loss_smooth_epoch,
        )
