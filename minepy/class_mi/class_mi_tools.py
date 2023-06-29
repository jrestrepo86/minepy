#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification mutual information tools

"""


import numpy as np
import torch
import torch.nn as nn

from minepy.minepy_tools import get_activation_fn, toColVector

EPS = 1e-6


def class_mi_batch(x, z, batch_size=1, shuffle=True):
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
        x_b = x[i * batch_size : (i + 1) * batch_size]
        z_b = z[i * batch_size : (i + 1) * batch_size]

        batches.append((x_b, z_b))

    return batches


class ClassMiModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, afn):
        super().__init__()
        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(num_hidden_layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, x, z):
        n = x.shape[0]
        x_marg = x[torch.randperm(n)]
        z_marg = z[torch.randperm(n)]

        # samples from joint distribution
        s_joint = torch.cat((x, z), dim=1)
        # samples from product of marginal distribution
        s_marg = torch.cat((x_marg, z_marg), dim=1)

        samples = torch.cat((s_joint, s_marg), dim=0)
        labels = torch.cat((torch.ones(n), torch.zeros(n)), dim=0).to(x.device)

        # random ordering
        inds = torch.randperm(2 * n)
        samples = samples[inds, :]
        labels = labels[inds]

        logit = torch.squeeze(self.model(samples))
        probs = torch.sigmoid(logit)

        return logit, labels, probs
