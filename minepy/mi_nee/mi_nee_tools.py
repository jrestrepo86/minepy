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


def hnee_data_loader(X, val_size=0.2, device="cuda"):
    n = X.shape[0]
    # send data top device
    X = torch.from_numpy(X)

    # mix samples
    inds = np.random.permutation(n)
    X = X[inds, :].to(device)
    # split data in training and validation sets
    val_size = int(val_size * n)
    inds = torch.randperm(n)
    (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

    Xtrain = X[train_idx, :]
    Xval = X[val_idx, :]
    Xtrain = Xtrain.to(device)
    Xval = Xval.to(device)
    return Xtrain, Xval, X
