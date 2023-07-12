#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification mutual information tools

"""


import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from minepy.minepy_tools import get_activation_fn, toColVector

EPS = 1e-6


def class_mi_batch(x, z, batch_size=1, shuffle=True):
    n = len(x)
    device = x.get_device()
    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm].to(device)
        z = z[rand_perm].to(device)
    x_marg = x[torch.randperm(n)].to(device)
    z_marg = z[torch.randperm(n)].to(device)

    batches = []
    for i in range(n // batch_size):
        inds = np.arange(i * batch_size, (i + 1) * batch_size, dtype=int)
        batches.append(
            (
                x[inds],
                z[inds],
                x_marg[inds],
                z_marg[inds],
            )
        )

    return batches


def class_cmi_batch(x, y, z, k=1, batch_size=1, shuffle=True):
    device = x.get_device()
    # Gerate y|z samples
    x, y, z, y_cz = get_ycondz(x, y, z, k=1)
    n = x.shape[0]
    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm, :].to(device)
        y = y[rand_perm, :].to(device)
        z = z[rand_perm, :].to(device)
        y_cz = y_cz[rand_perm, :].to(device)

    batches = []
    for i in range(n // batch_size):
        inds = np.arange(i * batch_size, (i + 1) * batch_size, dtype=int)
        batches.append(
            (
                x[inds],
                z[inds],
                y[inds],
                y_cz[inds],
            )
        )
    return batches


def knn(x, k=1, p=2):
    dist = distance_matrix(x, x, p)
    knn = dist.topk(k + 1, largest=False)
    return torch.squeeze(knn.indices[:, 1:])


def distance_matrix(x, y, p=2):  # pairwise distance of vectors
    n = x.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, n, d)
    y = y.unsqueeze(0).expand(n, n, d)

    dist = torch.pow(x - y, p).sum(2) ** (1 / p)

    return dist


def get_ycondz(x, y, z, k=1):
    # split samples in normal and marginal
    split_ind = int(x.shape[0] / 2)
    ztrain = z[:split_ind, :]
    ytrain = y[:split_ind, :]
    xtest = x[split_ind:, :]
    ytest = y[split_ind:, :]
    ztest = z[split_ind:, :]

    # Gerate y|z samples (knn)
    nbrs = NearestNeighbors(n_neighbors=k).fit(ztrain.cpu())
    indx = nbrs.kneighbors(ztest.cpu(), return_distance=False).flatten()
    y_cz = ytrain[indx, :]
    return xtest, ytest, ztest, y_cz


def split_data_train_val(X, Y, Z, val_size, device):
    X = torch.from_numpy(toColVector(X.astype(np.float32)))
    Y = torch.from_numpy(toColVector(Y.astype(np.float32)))
    Z = torch.from_numpy(toColVector(Z.astype(np.float32)))

    N, _ = X.shape

    val_size = int(val_size * N)
    inds = np.random.permutation(N)
    (
        val_idx,
        train_idx,
    ) = (
        inds[:val_size],
        inds[val_size:],
    )
    Xval, Xtrain = X[val_idx, :], X[train_idx, :]
    Yval, Ytrain = Y[val_idx, :], Y[train_idx, :]
    Zval, Ztrain = Z[val_idx, :], Z[train_idx, :]

    Xval = Xval.to(device)
    Xtrain = Xtrain.to(device)
    Yval = Xval.to(device)
    Ytrain = Xtrain.to(device)
    Zval = Zval.to(device)
    Ztrain = Ztrain.to(device)

    Xtest = X.to(device)
    Ytest = Y.to(device)
    Ztest = Z.to(device)

    return Xtrain, Ytrain, Ztrain, Xval, Yval, Zval, Xtest, Ytest, Ztest
