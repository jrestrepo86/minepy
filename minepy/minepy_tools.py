#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minepy Tools

"""

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
