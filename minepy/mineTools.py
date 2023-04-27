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


def MIbatch(x, z, batchSize=1, shuffle=True):

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
    for i in range(n // batchSize):
        x_b = x[i * batchSize:(i + 1) * batchSize]
        z_b = z[i * batchSize:(i + 1) * batchSize]

        batches.append((x_b, z_b))

    return batches


class EarlyStopping:

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def TEbatch(source,
            target,
            m=1,
            tau=1,
            u=1,
            batchSize=1,
            shuffle=True,
            fullBatch=False):

    # embedding
    target = embedding(target, m, tau)
    source = embedding(source, m, tau)
    tu = target[u:, :]
    tm = target[:-u, :]
    sm = source[:-u, :]
    tu = torch.from_numpy(tu).float()
    tm = torch.from_numpy(tm).float()
    sm = torch.from_numpy(sm).float()

    n = tu.shape[0]

    if shuffle:
        rand_perm = torch.randperm(n)
        tu = tu[rand_perm, :]
        tm = tm[rand_perm, :]
        sm = sm[rand_perm, :]

    if fullBatch:
        return (tu, tm, sm)
    else:
        batches = []
        for i in range(n // batchSize):
            tu_b = tu[i * batchSize:(i + 1) * batchSize, :]
            tm_b = tm[i * batchSize:(i + 1) * batchSize, :]
            sm_b = sm[i * batchSize:(i + 1) * batchSize, :]

            batches.append((tu_b, tm_b, sm_b))
        return batches


def TEbatch02(data, batchSize=1, shuffle=True, fullBatch=False):

    n = data.shape[0]

    if shuffle:
        rand_perm = torch.randperm(n)
        data = data[rand_perm, :]

    if fullBatch:
        return (data)
    else:
        batches = []
        for i in range(n // batchSize):
            batches.append(data[i * batchSize:(i + 1) * batchSize, :])
        return batches


def Interp(x, ntimes):
    n = x.size
    t = np.linspace(0, n, n)
    f = interp1d(t, x, kind=2, copy=True, fill_value=(x[0], x[-1]))
    tnew = np.linspace(0, n, ntimes * n)
    return f(tnew)


def henon(n, c):
    """Coupled Henon map X1->X2"""
    n0 = 8000
    n = n + n0
    x = np.zeros((n, 2))
    x[0:2, :] = np.random.rand(2, 2) * 0.1
    for i in range(1, n):
        x[i, 0] = 1.4 - x[i - 1, 0]**2 + 0.3 * x[i - 2, 0]

        x[i, 1] = (1.4 - c * x[i - 1, 0] * x[i - 1, 1] -
                   (1 - c) * x[i - 1, 1]**2 + 0.3 * x[i - 2, 1])

    return x[n0:, :]
