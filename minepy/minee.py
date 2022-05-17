#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import minepy.mineTools as mineTools
from minepy.mineLayers import customNet


class Minee(nn.Module):

    def __init__(self,
                 NetX=None,
                 NetZ=None,
                 NetXZ=None,
                 input_dim=2,
                 afn='relu',
                 hidden_dim=50,
                 nLayers=2,
                 device=None):

        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        torch.device(self.device)
        self.epochMI = []
        self.epochHx = []
        self.epochHz = []
        self.epochHxz = []
        if NetX is None:
            NetX = customNet(input_dim=input_dim,
                             hidden_dim=hidden_dim,
                             afn=afn,
                             nLayers=nLayers)
        if NetZ is None:
            NetZ = customNet(input_dim=input_dim,
                             hidden_dim=hidden_dim,
                             afn=afn,
                             nLayers=nLayers)
        if NetXZ is None:
            NetXZ = customNet(input_dim=2 * input_dim,
                              hidden_dim=hidden_dim,
                              afn=afn,
                              nLayers=nLayers)

        self.NetX = NetX.to(self.device)
        self.NetZ = NetZ.to(self.device)
        self.NetXZ = NetXZ.to(self.device)

    def uniformSample(self, x, z):

        max_x, _ = torch.max(x, dim=0, keepdim=True)
        min_x, _ = torch.min(x, dim=0, keepdim=True)
        max_z, _ = torch.max(z, dim=0, keepdim=True)
        min_z, _ = torch.min(z, dim=0, keepdim=True)

        ux = (max_x - min_x) * torch.rand(
            (self.ref_batchSize, x.shape[1]), device=self.device) + min_x
        uz = (max_z - min_z) * torch.rand(
            (self.ref_batchSize, z.shape[1]), device=self.device) + min_z
        return ux, uz

    def div(self, Net, data, ref):
        mean_f = Net(data).mean()
        log_mean_ef_ref = torch.logsumexp(Net(ref), 0) - np.log(ref.shape[0])
        return mean_f - log_mean_ef_ref

    def forward(self, x, z):
        ux, uz = self.uniformSample(x, z)
        loss_x = -self.div(self.NetX, x, ux)
        loss_z = -self.div(self.NetZ, z, uz)
        loss_xz = -self.div(self.NetXZ, torch.cat(
            (x, z), dim=1), torch.cat((ux, uz), dim=1))

        return loss_x, loss_z, loss_xz

    def optimize(self,
                 X,
                 Z,
                 batchSize,
                 numEpochs,
                 batchScale=1,
                 optimizer=None,
                 lr=1e-4,
                 disableTqdm=True):
        if optimizer is None:
            opt_x = torch.optim.Adam(self.NetX.parameters(),
                                     lr=lr,
                                     betas=(0.5, 0.999))
            opt_z = torch.optim.Adam(self.NetZ.parameters(),
                                     lr=lr,
                                     betas=(0.5, 0.999))
            opt_xz = torch.optim.Adam(self.NetXZ.parameters(),
                                      lr=lr,
                                      betas=(0.5, 0.999))

        self.NetX.train()  # Set model to training mode
        self.NetZ.train()
        self.NetXZ.train()

        self.ref_batchSize = int(batchSize * batchScale)

        for epoch in tqdm(range(numEpochs), disable=disableTqdm):
            mi, hx, hz, hxz = 0, 0, 0, 0
            for x, z in mineTools.MIbatch(X, Z, batchSize):
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

        Hx, Hz, Hxz, MI = self.getMI(X, Z)
        vals = (Hx, Hz, Hxz, MI)
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
        for layer in self.NetX.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.NetZ.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.NetXZ.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
