#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


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


class MineNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=100, afn='relu', nLayers=2):
        super().__init__()
        self.name = 'MineNet'
        activation_fn = get_activation_fn(afn)
        # final_activation_fn = get_activation_fn('elu')
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(nLayers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        return self.layers(x)


class T1(nn.Module):

    def __init__(self, input_dim, dim_feedforward=100, dropout=0):
        super().__init__()
        self.name = 'T1'
        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Tenee(nn.Module):

    def __init__(self, input_dim, dim_feedforward=100, dropout=0.0):
        super().__init__()
        self.name = 'teneeNet'
        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(p=dropout),
            # nn.ReLU(),
            nn.ELU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.Dropout(p=dropout),
            # nn.ReLU(),
            nn.ELU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x):
        return self.layers(x)


class T11(nn.Module):

    def __init__(self, input_dim, dim_feedforward=100, dropout=0):
        super().__init__()
        self.name = 'T1'
        self.dim = 1
        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x):
        return self.layers(x)


class T2(nn.Module):

    def __init__(self, input_dim, dim_feedforward=100):
        # super(Net, self).__init__()
        super().__init__()
        self.name = 'T2'
        self.fc1 = nn.Linear(input_dim, dim_feedforward)
        self.fc2 = nn.Linear(input_dim, dim_feedforward)
        self.fc3 = nn.Linear(dim_feedforward, 1)

    def forward(self, x, z):
        return self.fc3(F.relu(self.fc1(x) + self.fc2(z)))


class T3(nn.Module):
    '''
        Linear Generator
        https://github.com/gtegner/mine-pytorch/blob/master/mine/models/layers.py
    '''

    def __init__(self, input_dim, output_dim=1, dim_feedforward=100):
        super().__init__()
        self.name = 'T3'
        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.BatchNorm1d(dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.ReLU(),
            nn.BatchNorm1d(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class T3p(nn.Module):
    '''
        Linear Generator
        https://github.com/gtegner/mine-pytorch/blob/master/mine/models/layers.py
    '''

    def __init__(self, input_dim, output_dim=1, dim_feedforward=100):
        super().__init__()
        self.name = 'T3'
        self.dim = 1
        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(dim_feedforward),
            nn.Linear(dim_feedforward, dim_feedforward),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim),
        )

    def forward(self, x, z):
        return self.layers(torch.cat((x, z), dim=self.dim))
