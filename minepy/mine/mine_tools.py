#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine Tools

"""

import numpy as np
import torch
from matplotlib.pyplot import margins

#
# class mine_data_loader:
#     def __init__(self, X, Y, val_size=0.2, device="cuda"):
#         self.set_joint_marginals(X, Y)
#         self.split_train_val(val_size, device)
#
#     def set_joint_marginals(self, X, Y):
#         n = X.shape[0]
#         # set marginals
#         Xmarg = X[np.random.permutation(n), :]
#         Ymarg = Y[np.random.permutation(n), :]
#         self.joint_samples = np.hstack((X, Y))
#         self.marg_samples = np.hstack((Xmarg, Ymarg))
#
#     def split_train_val(self, val_size, device):
#         n = self.joint_samples.shape[0]
#         # send data top device
#         samples = torch.from_numpy(self.samples.astype(np.float32))
#
#         # mix samples
#         inds = np.random.permutation(n)
#         samples = samples[inds, :]
#         # split data in training and validation sets
#         val_size = int(val_size * n)
#         inds = torch.randperm(n)
#         (val_idx, train_idx) = (inds[:val_size], inds[val_size:])
#
#         train_samples = samples[train_idx, :]
#
#         val_samples = samples[val_idx, :]
#         self.train_samples = train_samples.to(device)
#         self.val_samples = val_samples.to(device)


def mine_data_loader(X, Y, val_size=0.2, device="cuda"):
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


# def mine_data_loader(X, Y, val_size=0.2, device="cuda"):
#     n = X.shape[0]
#     # send data top device
#     X = torch.from_numpy(X)
#     Y = torch.from_numpy(Y)
#     # mix samples
#     inds = np.random.permutation(n)
#     X = X[inds, :]
#     Y = Y[inds, :]
#     # split data in training and validation sets
#     val_size = int(val_size * n)
#     inds = torch.randperm(n)
#     (val_idx, train_idx) = (inds[:val_size], inds[val_size:])
#
#     Xtrain = X[train_idx, :]
#     Ytrain = Y[train_idx, :]
#     Xval = X[val_idx, :]
#     Yval = Y[val_idx, :]
#     Xtrain = Xtrain
#     Ytrain = Ytrain
#     Xval = Xval
#     Yval = Yval
#     train_joint_samples = torch.cat((Xtrain, Ytrain), dim=1)
#     Ymarg = Ytrain[torch.randperm(Ytrain.shape[0]), :]
#     train_marg_samples = torch.cat((Xtrain, Ymarg), dim=1)
#     val_joint_samples = torch.cat((Xval, Yval), dim=1)
#     Ymarg = Yval[torch.randperm(Yval.shape[0]), :]
#     val_marg_samples = torch.cat((Xval, Ymarg), dim=1)
#
#     joint_samples = torch.cat((X, Y), dim=1)
#     Ymarg = Y[torch.randperm(Y.shape[0]), :]
#     marg_samples = torch.cat((X, Ymarg), dim=1)
#     return (
#         joint_samples.to(device),
#         marg_samples.to(device),
#         train_joint_samples.to(device),
#         train_marg_samples.to(device),
#         val_joint_samples.to(device),
#         val_marg_samples.to(device),
#     )


# def mine_data_loader(X, Y, val_size=0.2, device="cuda"):
#     n = X.shape[0]
#     # send data top device
#     X = torch.from_numpy(X.astype(np.float32))
#     Y = torch.from_numpy(Y.astype(np.float32))
#
#     samples = torch.cat((X, Y), dim=1)
#
#     # mix samples
#     inds = torch.randperm(n)
#     samples = samples[inds, :]
#     # split data in training and validation sets
#     val_size = int(val_size * n)
#     inds = torch.randperm(n)
#     (val_idx, train_idx) = (inds[:val_size], inds[val_size:])
#
#     train_samples = samples[train_idx, :]
#     val_samples = samples[val_idx, :]
#
#     return (
#         train_samples.to(device),
#         val_samples.to(device),
#     )


# Xtrain = X[train_idx, :]
# Ytrain = Y[train_idx, :]
# Xval = X[val_idx, :]
# Yval = Y[val_idx, :]
# Xtrain = Xtrain.to(device)
# Ytrain = Ytrain.to(device)
# Xval = Xval.to(device)
# Yval = Yval.to(device)
# return Xtrain, Ytrain, Xval, Yval, X, Y
