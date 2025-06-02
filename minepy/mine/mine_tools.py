#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine Tools

"""

import numpy as np
import torch


def mine_data_loader(X, Y, val_size=0.2, random_sample=True, device="cuda"):
    n = X.shape[0]
    # send data top device
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    val_size = int(val_size * n)
    if random_sample:
        # random sampling
        inds = np.random.permutation(n)
        X = X[inds, :]
        Y = Y[inds, :]
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
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xval, Yval = X[val_idx], Y[val_idx]
    Xtrain = Xtrain.to(device)
    Ytrain = Ytrain.to(device)
    Xval = Xval.to(device)
    Yval = Yval.to(device)
    X = X.to(device)
    Y = Y.to(device)
    return Xtrain, Ytrain, Xval, Yval, X, Y
