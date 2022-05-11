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
        self.regWeight = 1
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

    def div(self, Net, data, ref):
        # Calculate the divergence estimate using a neural network
        mean_f = Net(data).mean()
        log_mean_ef_ref = torch.logsumexp(Net(ref), 0) - np.log(ref.shape[0])
        return mean_f, log_mean_ef_ref

    def forward(self, Net, data):
        # utu, utm, usm = self.uniformSample(tu, tm, sm)
        udata = self.uniformSample(data)
        mean_f, log_mean_ef_ref = self.div(Net, data, udata)
        reg = self.regWeight * torch.pow(log_mean_ef_ref - self.targetVal, 2)
        loss = -(mean_f - log_mean_ef_ref) + reg
        return loss

    def train(self,
              target,
              source,
              batchSize,
              numEpochs,
              batchScale=1,
              disableTqdm=True):
        self.batchSize = batchSize
        self.numEpochs = numEpochs
        self.ref_batchSize = int(batchSize * batchScale)
        self.disableTqdm = disableTqdm
        opt1 = torch.optim.Adam(self.Net1.parameters(), lr=1e-4)
        opt2 = torch.optim.Adam(self.Net2.parameters(), lr=1e-4)
        opt3 = torch.optim.Adam(self.Net3.parameters(), lr=1e-4)
        opt4 = torch.optim.Adam(self.Net4.parameters(), lr=1e-4)

        d = self.m
        target = tools.embedding(target, d, self.tau)
        source = tools.embedding(source, d, self.tau)
        tu = target[self.u:, :]
        tm = target[:-self.u, :]
        sm = source[:-self.u, :]
        data = np.concatenate((tu, tm, sm), axis=1)

        d = self.m
        # H(tu,tm,sm)
        # self.regWeight = 3 * d
        # self.ref_batchSize = int(3 * d * batchSize * batchScale)
        self.targetVal = 3
        self.epochN1 = self.optimize(data, self.Net1, opt1)
        t1 = self.evalTenee(data, self.Net1)
        # H(tm,sm)
        # self.regWeight = 2 * d
        # self.ref_batchSize = int(2 * d * batchSize * batchScale)
        self.targetVal = 2
        self.epochN2 = self.optimize(data[:, d:], self.Net2, opt2)
        t2 = self.evalTenee(data[:, d:], self.Net2)
        # H(tu,tm)
        # self.regWeight = 2 * d
        # self.ref_batchSize = int(2 * d * batchSize * batchScale)
        self.targetVal = 2
        self.epochN3 = self.optimize(data[:, :2 * d], self.Net3, opt3)
        t3 = self.evalTenee(data[:, :2 * d], self.Net3)
        # H(tm)
        # self.regWeight = d
        self.targetVal = 1
        # self.ref_batchSize = int(d * batchSize * batchScale)
        self.epochN4 = self.optimize(data[:, d:2 * d], self.Net4, opt4)
        t4 = self.evalTenee(data[:, d:2 * d], self.Net4)

        # Epoch TE
        self.epochTE = self.epochN2 - self.epochN1 + self.epochN3 - self.epochN4
        self.TE = t2 - t1 + t3 - t4
        epochArray = (self.epochTE, self.epochN1, self.epochN2, self.epochN3,
                      self.epochN4)

        return self.TE, epochArray

    def optimize(self, data, Net, opt):
        Net.train()  # Set model to training mode
        h = []
        for epoch in tqdm(range(self.numEpochs), disable=self.disableTqdm):
            for d in tools.TEbatch02(data, batchSize=self.batchSize):
                d = torch.from_numpy(d).float().to(self.device)
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    loss = self.forward(Net, d)
                    loss.backward()
                    opt.step()
            h.append(-loss.item())

        return np.array(h)

    def evalTenee(self, data, Net):
        data = torch.from_numpy(data).float().to(self.device)
        # H(tu,tm,sm)
        with torch.no_grad():
            mean_f, log_mean_ef_ref = self.div(Net, data,
                                               self.uniformSample(data))
        reg = self.regWeight * torch.pow(log_mean_ef_ref - self.targetVal, 2)
        h = mean_f - log_mean_ef_ref + reg
        return h.item()

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
