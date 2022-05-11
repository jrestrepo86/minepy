#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import minepy.mineTools as mineTools


class Minee(nn.Module):

    def __init__(self, Tx, Tz, Txz, loss=None, alpha=0.01, device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        torch.device(self.device)
        self.Tx = Tx.to(self.device)
        self.Tz = Tz.to(self.device)
        self.Txz = Txz.to(self.device)
        self.loss = loss
        self.alpha = alpha
        self.MI = -20
        self.Hx = -20
        self.Hz = -20
        self.Hxz = -20
        self.epochMI = []
        self.epochHx = []
        self.epochHz = []
        self.epochHxz = []

    def uniformSample(self, x, z):

        max_x = x.max()
        min_x = x.min()
        max_z = z.max()
        min_z = z.min()

        ux = (max_x - min_x) * torch.rand(
            (self.ref_batchSize, x.shape[1]), device=self.device) + min_x
        uz = (max_z - min_z) * torch.rand(
            (self.ref_batchSize, z.shape[1]), device=self.device) + min_z
        uxz = torch.cat((ux, uz), dim=1)
        return ux, uz, uxz

    def div(self, T, data, ref):
        # Calculate the divergence estimate using a neural network
        mean_f = T(data).mean()
        log_mean_ef_ref = torch.logsumexp(T(ref), 0) - np.log(ref.shape[0])
        return mean_f - log_mean_ef_ref

    def forward(self, x, z):
        ux, uz, uxz = self.uniformSample(x, z)
        loss_x = -self.div(self.Tx, x, ux)
        loss_z = -self.div(self.Tz, z, uz)
        loss_xz = -self.div(self.Txz, torch.cat((x, z), dim=1), uxz)

        return loss_x, loss_z, loss_xz

    def optimize(self,
                 X,
                 Z,
                 batchSize,
                 numEpochs,
                 batchScale=1,
                 optimizer=None,
                 disableTqdm=True):
        if optimizer is None:
            opt_x = torch.optim.Adam(self.Tx.parameters(), lr=1e-4)
            opt_z = torch.optim.Adam(self.Tz.parameters(), lr=1e-4)
            opt_xz = torch.optim.Adam(self.Txz.parameters(), lr=1e-4)

        self.Tx.train()  # Set model to training mode
        self.Tz.train()
        self.Txz.train()

        self.ref_batchSize = int(batchSize * batchScale)

        for epoch in tqdm(range(numEpochs), disable=disableTqdm):
            mi, hx, hz, hxz = 0, 0, 0, 0
            for x, z in mineTools.batch(X, Z, batchSize):
                x = x.to(self.device)
                z = z.to(self.device)
                opt_x.zero_grad()
                opt_z.zero_grad()
                opt_xz.zero_grad()
                with torch.set_grad_enabled(True):
                    loss_x, loss_z, loss_xz = self.forward(x, z)
                    loss_x.backward()
                    loss_z.backward()
                    loss_xz.backward()
                    opt_x.step()
                    opt_z.step()
                    opt_xz.step()
                    hx, hz, hxz = -loss_x.item(), -loss_z.item(
                    ), -loss_xz.item()
                    mi = hxz - hx - hz
            self.epochHx.append(hx)
            self.epochHz.append(hz)
            self.epochHxz.append(hxz)
            self.epochMI.append(mi)

        self.Hx, self.Hz, self.Hxz, self.MI = self.getMI(X, Z)
        vals = (self.Hx, self.Hz, self.Hxz, self.MI)
        EpochVals = (np.array(self.epochHx), np.array(self.epochHz),
                     np.array(self.epochHxz), np.array(self.epochMI))

        return vals, EpochVals

    def getMI(self, x, z):
        if isinstance(x, np.ndarray):
            x = mineTools.toColVector(x)
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = mineTools.toColVector(z)
            z = torch.from_numpy(z).float()

        x = x.to(self.device)
        z = z.to(self.device)
        with torch.no_grad():
            hx, hz, hxz = self.forward(x, z)
        hx = -hx.item()
        hz = -hz.item()
        hxz = -hxz.item()
        return hx, hz, hxz, hxz - hx - hz

    def netReset(self):
        for layer in self.Tx.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.Tz.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.Txz.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
