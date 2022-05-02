#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


class MineNet(nn.Module):

    def __init__(self, input_dim, dim_feedforward=100, rect='relu', dropout=0):
        super().__init__()
        self.name = 'MineNet'
        if rect == 'elu':
            Rect1 = nn.ELU()
            Rect2 = nn.ELU()
        elif rect == 'leakyRelu':
            Rect1 = nn.LeakyReLU()
            Rect2 = nn.LeakyReLU()
        else:
            Rect1 = nn.ReLU()
            Rect2 = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(p=dropout),
            Rect1,
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.Dropout(p=dropout),
            Rect2,
            nn.Linear(dim_feedforward, 1),
        )

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
