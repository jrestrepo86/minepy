#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minepy Tools

"""

import numpy as np
import torch
from scipy.interpolate import interp1d


def toColVector(x):
    """
    Change vectors to colum vectors
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def batch(x, z, batchSize=1, shuffle=True):

    if isinstance(x, np.ndarray):
        x = toColVector(x)
        x = torch.from_numpy(x).float()
    if isinstance(z, np.ndarray):
        z = toColVector(z)
        z = torch.from_numpy(z).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        z = z[rand_perm]

    batches = []
    for i in range(n // batchSize):
        x_b = x[i * batchSize:(i + 1) * batchSize]
        z_b = z[i * batchSize:(i + 1) * batchSize]

        batches.append((x_b, z_b))

    return batches


def Interp(x, ntimes):
    n = x.size
    t = np.linspace(0, n, n)
    f = interp1d(t, x, kind=2, copy=True, fill_value=(x[0], x[-1]))
    tnew = np.linspace(0, n, ntimes * n)
    return f(tnew)
