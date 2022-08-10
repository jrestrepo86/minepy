#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import minepy.mineTools as mineTools
from minepy.mineLayers import MineNet

EPS = 1e-6


class Clamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


class EMALoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema
    return t_log, running_mean


class Mine(nn.Module):

    def __init__(self,
                 Net=None,
                 input_dim=2,
                 loss='mine_biased',
                 afn='relu',
                 hidden_dim=50,
                 nLayers=2,
                 alpha=0.01,
                 regWeight=1.0,
                 targetVal=0.0,
                 clip=1.0,
                 device=None):
        super().__init__()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        torch.device(self.device)
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.regWeight = regWeight
        self.targetVal = targetVal
        self.clip = clip
        self.epochMI = []

        if Net is None:
            Net = MineNet(
                input_dim,
                hidden_dim=hidden_dim,
                afn=afn,
                nLayers=nLayers,
            )
        self.Net = Net.to(self.device)

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(z.shape[0])]

        t = self.Net(x, z).mean()
        t_marg = self.Net(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg,
                                                      self.running_mean,
                                                      self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ["remine"]:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])
            second_term += self.regWeight * torch.pow(
                second_term - self.targetVal, 2)
        elif self.loss in ['clip']:
            # with torch.no_grad():
            # t_marg = Clamp.apply(t_marg, -self.clip, self.clip)
            t_marg = torch.clamp(t_marg, min=-self.clip, max=self.clip)
            # print(f'{zc.to("cpu").detach().numpy()}')
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])
        else:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])  # mine_biased default

        return -t + second_term

    def optimize(self,
                 X,
                 Z,
                 batchSize,
                 numEpochs,
                 opt=None,
                 lr=1e-4,
                 disableTqdm=True):

        if opt is None:
            opt = torch.optim.Adam(self.Net.parameters(),
                                   lr=lr,
                                   betas=(0.5, 0.999))

        self.train()  # Set model to training mode
        for _ in tqdm(range(numEpochs), disable=disableTqdm):
            mu_mi = 0
            for x, z in mineTools.MIbatch(X, Z, batchSize):
                x = x.to(self.device)
                z = z.to(self.device)
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    loss = self.forward(x, z)
                    loss.backward()
                    opt.step()
                    mu_mi = -loss.item()
            self.epochMI.append(mu_mi)

        MI = self.evalMI(X, Z)
        return MI, np.array(self.epochMI)

    def evalMI(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = mineTools.toColVector(x)
            x = torch.from_numpy(x.copy()).float()
        if isinstance(z, np.ndarray):
            z = mineTools.toColVector(z)
            z = torch.from_numpy(z.copy()).float()
        # self.clip = None
        x = x.to(self.device)
        z = z.to(self.device)
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi.item()

    def netReset(self):
        for layer in self.Net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
