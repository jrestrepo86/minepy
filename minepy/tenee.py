#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import minepy.mineLayers as Layers
import minepy.mineTools as tools


class Tenee(nn.Module):

    def __init__(self, m=1, tau=1, u=1, dim_feedforward=50, device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        torch.device(self.device)
        self.m = m
        self.tau = tau
        self.u = u
        self.Net(input_dim=self.m, dim_feedforward=dim_feedforward)
        self.regWeight = 0
        self.targetVal = 0
        self.TE = -20
        self.epochTE = []
        self.epochN1 = []
        self.epochN2 = []
        self.epochN3 = []
        self.epochN4 = []

    def Net(self, input_dim, dim_feedforward):
        # H(tu, tm, sm)
        self.Net1 = Layers.Tenee(3 * input_dim,
                                 dim_feedforward=dim_feedforward).to(
                                     self.device)
        # H(tm, sm)
        self.Net2 = Layers.Tenee(2 * input_dim,
                                 dim_feedforward=dim_feedforward).to(
                                     self.device)
        # H(tu, tm)
        self.Net3 = Layers.Tenee(2 * input_dim,
                                 dim_feedforward=dim_feedforward).to(
                                     self.device)
        # H(tm)
        self.Net4 = Layers.Tenee(
            input_dim, dim_feedforward=dim_feedforward).to(self.device)

    def uniformSample(self, data):
        udata = torch.rand((self.ref_batchSize, data.shape[1]),
                           device=self.device)
        for i, col in enumerate(data.t()):
            max_ = col.max()
            min_ = col.min()
            udata[:, i] = (max_ - min_) * udata[:, i] + min_
        return udata

    def div(self, T, data, ref):
        # Calculate the divergence estimate using a neural network
        mean_f = T(data).mean()
        log_mean_ef_ref = torch.logsumexp(T(ref), 0) - np.log(ref.shape[0])
        reg = self.regWeight * torch.pow(log_mean_ef_ref - self.targetVal, 2)
        return mean_f - log_mean_ef_ref - reg

    def forward(self, tu, tm, sm):
        # utu, utm, usm = self.uniformSample(tu, tm, sm)
        data = torch.cat((tu, tm, sm), dim=1)
        udata = self.uniformSample(data)
        d = self.m

        # H(tu,tm, sm)
        loss1 = -self.div(self.Net1, data, udata)
        # H(tm, sm)
        loss2 = -self.div(self.Net2, data[:, d:], udata[:, d:])
        # H(tu, tm)
        loss3 = -self.div(self.Net3, data[:, :2 * d], udata[:, :2 * d])
        # H(tm)
        loss4 = -self.div(self.Net4, data[:, d:2 * d], udata[:, d:2 * d])

        return loss1, loss2, loss3, loss4

    def optimize(self,
                 target,
                 source,
                 batchSize,
                 numEpochs,
                 batchScale=1,
                 optimizer=None,
                 disableTqdm=True):
        if optimizer is None:
            opt1 = torch.optim.Adam(self.Net1.parameters(), lr=1e-4)
            opt2 = torch.optim.Adam(self.Net2.parameters(), lr=1e-4)
            opt3 = torch.optim.Adam(self.Net3.parameters(), lr=1e-4)
            opt4 = torch.optim.Adam(self.Net4.parameters(), lr=1e-4)

        self.Net1.train()  # Set model to training mode
        self.Net2.train()
        self.Net3.train()
        self.Net4.train()

        self.ref_batchSize = int(batchSize * batchScale)

        for epoch in tqdm(range(numEpochs), disable=disableTqdm):
            te, t1, t2, t3, t4 = 0, 0, 0, 0, 0
            for tu, tm, sm in tools.TEbatch(target,
                                            source,
                                            m=self.m,
                                            tau=self.tau,
                                            u=self.u,
                                            batchSize=batchSize):
                tu = tu.to(self.device)
                tm = tm.to(self.device)
                sm = sm.to(self.device)
                opt1.zero_grad()
                opt2.zero_grad()
                opt3.zero_grad()
                opt4.zero_grad()
                with torch.set_grad_enabled(True):
                    loss1, loss2, loss3, loss4 = self.forward(tu, tm, sm)

                    loss1.backward()
                    opt1.step()

                    loss2.backward()
                    opt2.step()

                    loss3.backward()
                    opt3.step()

                    loss4.backward()
                    opt4.step()
                    t1, t2, t3, t4 = -loss1.item(), -loss2.item(), -loss3.item(
                    ), -loss4.item()
                    te = t2 - t1 + t3 - t4
            self.epochTE.append(te)
            self.epochN1.append(t1)
            self.epochN2.append(t2)
            self.epochN3.append(t3)
            self.epochN4.append(t4)

        self.TE = self.getMI(target, source)
        epochArray = (np.array(self.epochTE), np.array(self.epochN1),
                      np.array(self.epochN2), np.array(self.epochN3),
                      np.array(self.epochN4))
        return self.TE, epochArray

    def getMI(self, target, source):

        (tu, tm, sm) = tools.TEbatch(target,
                                     source,
                                     self.m,
                                     self.tau,
                                     self.u,
                                     fullBatch=True)
        tu = tu.to(self.device)
        tm = tm.to(self.device)
        sm = sm.to(self.device)
        with torch.no_grad():
            data = torch.cat((tu, tm, sm), dim=1)
            udata = self.uniformSample(data)
            d = self.m

            # H(tu,tm, sm)
            loss1 = self.div(self.Net1, data, udata)
            # H(tm, sm)
            loss2 = self.div(self.Net2, data[:, d:], udata[:, d:])
            # H(tu, tm)
            loss3 = self.div(self.Net3, data[:, :2 * d], udata[:, :2 * d])
            # H(tm)
            loss4 = self.div(self.Net4, data[:, d:2 * d], udata[:, d:2 * d])
            t1 = loss1.item()
            t2 = loss2.item()
            t3 = loss3.item()
            t4 = loss4.item()
        return t2 - t1 + t3 - t4

    def netReset(self):
        for layer in self.Net1.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.Net2.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.Net3.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.Net4.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
