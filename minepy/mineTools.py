#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minepy Tools

"""

import math

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6


def toColVector(x):
    """
    Change vectors to column vectors
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    x.reshape((-1, 1))
    return x


def embedding(x, m, tau):
    """ Time series embedding """
    x = toColVector(x)
    n, k = x.shape
    l_ = n - (m - 1) * tau
    V = np.zeros((l_, m * k))

    for j in range(k):
        for i in range(m):
            V[:, i + j * m] = x[i * tau:i * tau + l_, j]
    return V


def get_activation_fn(afn):
    activation_functions = {
        "linear": lambda: lambda x: x,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "leaky_relu": nn.LeakyReLU,
        "threshold": nn.Threshold,
        "hardtanh": nn.Hardtanh,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "log_sigmoid": nn.LogSigmoid,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
    }

    if afn not in activation_functions:
        raise ValueError(
            f"'{afn}' is not included in activation_functions. Use below one \n {activation_functions.keys()}"
        )

    return activation_functions[afn]


def mine_batch(x, z, batch_size=1, shuffle=True):

    if isinstance(x, np.ndarray):
        x = toColVector(x)
        x = torch.from_numpy(x.copy()).float()
    if isinstance(z, np.ndarray):
        z = toColVector(z)
        z = torch.from_numpy(z.copy()).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        z = z[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size:(i + 1) * batch_size]
        z_b = z[i * batch_size:(i + 1) * batch_size]

        batches.append((x_b, z_b))

    return batches


class EarlyStopping:

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = np.abs(delta)
        self.counter = 0
        self.early_stop = False
        self.min_loss = np.inf

    def __call__(self, loss):
        if torch.abs(loss - self.min_loss) <= self.delta:
            self.counter += 1
        else:
            if loss < self.min_loss:
                self.min_loss = loss
                self.counter = 0
            else:
                self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


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


class MineModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, afn, num_hidden_layers, loss,
                 alpha, regWeight, targetVal):
        super().__init__()
        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(num_hidden_layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*seq)
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.regWeight = regWeight
        self.targetVal = targetVal

    def forward(self, x, z):
        z_marg = z[torch.randperm(z.shape[0])]

        t = self.model(torch.cat((x, z), dim=1)).mean()
        t_marg = self.model(torch.cat((x, z_marg), dim=1))
        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg,
                                                      self.running_mean,
                                                      self.alpha)

            mi = t - second_term
            loss = -mi
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
            mi = t - second_term
            loss = -mi
        elif self.loss in ["remine"]:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])

            mi = t - second_term
            loss = -mi + self.regWeight * torch.pow(
                second_term - self.targetVal, 2)
        else:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])  # mine_biased as default
            mi = t - second_term
            loss = -mi

        return loss, mi
