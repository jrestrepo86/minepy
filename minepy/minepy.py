#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import minepy.mineTools as mineTools

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)


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

    def __init__(self, T, loss='mine', alpha=0.01, regWeight=2):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.MI = -20
        self.epochMI = []
        self.regWeight = regWeight
        self.targetVal = 0
        self.T = T.to(device)

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg,
                                                      self.running_mean,
                                                      self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ["mine_biased"]:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])
        elif self.loss in ["remine"]:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])
            second_term += self.regWeight * torch.pow(
                second_term - self.targetVal, 2)

        return -t + second_term

    def optimize(self, X, Z, batchSize, numEpochs, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        self.train()  # Set model to training mode
        for epoch in tqdm(range(numEpochs)):
            mu_mi = 0
            for x, z in mineTools.batch(X, Z, batchSize):
                x = x.to(device)
                z = z.to(device)
                opt.zero_grad()
                with torch.set_grad_enabled(True):
                    loss = self.forward(x, z)
                    loss.backward()
                    opt.step()
                    mu_mi = -loss.item()
            self.epochMI.append(mu_mi)

            # if epoch % (numEpochs // 3) == 0:
            #     print(f"It {epoch} - MI: {self.trainingMI}")

        self.MI = self.getMI(X, Z)
        print(f"Training MI: {self.MI}")
        return self.MI, self.epochMI

    def getMI(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = mineTools.toColVector(x)
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = mineTools.toColVector(z)
            z = torch.from_numpy(z).float()

        x = x.to(device)
        z = z.to(device)
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi.item()

    def netReset(self):
        for layer in self.T.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
