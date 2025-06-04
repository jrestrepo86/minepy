#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minee Tools

"""

import numpy as np
import torch


def minee_data_loader(X, Y, val_size=0.2, device="cuda"):
    n = X.shape[0]
    # send data top device
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    # mix samples
    inds = np.random.permutation(n)
    X = X[inds, :].to(device)
    Y = Y[inds, :].to(device)
    # split data in training and validation sets
    val_size = int(val_size * n)
    inds = torch.randperm(n)
    (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

    Xtrain = X[train_idx, :]
    Ytrain = Y[train_idx, :]
    Xval = X[val_idx, :]
    Yval = Y[val_idx, :]
    Xtrain = Xtrain.to(device)
    Ytrain = Ytrain.to(device)
    Xval = Xval.to(device)
    Yval = Yval.to(device)
    return Xtrain, Ytrain, Xval, Yval, X, Y


def hnee_data_loader(X, val_size=0.2, random_sample=True, device="cuda"):
    n = X.shape[0]
    # send data top device
    X = torch.from_numpy(X)

    val_size = int(val_size * n)
    if random_sample:
        # random sampling
        inds = np.random.permutation(n)
        X = X[inds, :]
        # split data in training and validation sets
        val_idx = torch.arange(val_size)
        train_idx = torch.arange(val_size, n)
    else:
        # Select contiguous validation samples starting from a random index
        start = torch.randint(0, n - val_size + 1, (1,)).item()
        val_idx = torch.arange(start, start + val_size)
        # Combine indices before start and after the validation block for training
        train_idx = torch.cat(
            [torch.arange(0, start), torch.arange(start + val_size, n)]
        )
    # Extract training and validation sets
    Xtrain = X[train_idx]
    Xval = X[val_idx]
    Xtrain = Xtrain.to(device)
    Xval = Xval.to(device)
    X = X.to(device)
    return Xtrain, Xval, X
