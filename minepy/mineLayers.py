#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


class ConcatLayer(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, z):
        return torch.cat((x, z), self.dim)


class T1(nn.Module):

    def __init__(self, input_dim, dim_feedforward=100, dropout=0):
        super().__init__()
        self.name = 'T1'
        self.dim = 1
        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x, z):
        return self.layers(torch.cat((x, z), dim=self.dim))


class T2(nn.Module):

    def __init__(self, dim_feedforward=100):
        # super(Net, self).__init__()
        super().__init__()
        self.name = 'T2'
        self.fc1 = nn.Linear(1, dim_feedforward)
        self.fc2 = nn.Linear(1, dim_feedforward)
        self.fc3 = nn.Linear(dim_feedforward, 1)

    def forward(self, x, z):
        h1 = F.relu(self.fc1(x) + self.fc2(z))
        h2 = self.fc3(h1)
        return h2


class T3(nn.Module):
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
            nn.BatchNorm1d(dim_feedforward),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim_feedforward),
            nn.Linear(dim_feedforward, output_dim),
        )

    def forward(self, x, z):
        return self.layers(torch.cat((x, z), dim=self.dim))
