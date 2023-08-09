#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification mutual information tools
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.pyplot import cla
from sklearn.neighbors import NearestNeighbors

EPS = 1e-6


class class_mi_data_loader:
    def __init__(self, X, Y, Z=None, val_size=0.2, device="cuda"):
        self.N, self.dx = X.shape
        self.dy = Y.shape[1]
        if Z is None:
            self.dz = False
        else:
            self.dz = Z.shape[1]
        self.X = X
        self.Y = Y
        self.Z = Z
        self.val_size = val_size
        self.device = device
        self.set_joint_marginals()
        self.split_train_val()

    def set_joint_marginals(self):
        n = self.N
        # set marginals
        X_marg = self.X[np.random.permutation(n), :]
        Y_marg = self.Y[np.random.permutation(n), :]
        if self.dz:
            Z_marg = self.Z[np.random.permutation(n), :]
            data_joint = np.hstack((self.X, self.Y, self.Z))
            data_marg = np.hstack((X_marg, Y_marg, Z_marg))
        else:
            data_joint = np.hstack((self.X, self.Y))
            data_marg = np.hstack((X_marg, Y_marg))

        self.samples = np.vstack((data_joint, data_marg))
        self.labels = np.squeeze(np.vstack((np.ones((n, 1)), np.zeros((n, 1)))))

    def split_train_val(self):
        n = self.samples.shape[0]
        self.train = {}
        self.val = {}
        # send data top device
        self.samples = torch.from_numpy(self.samples).to(self.device)
        self.labels = torch.from_numpy(self.labels).to(self.device)

        # mix samples
        inds = np.random.permutation(n)
        self.samples = self.samples[inds, :].to(self.device)
        self.labels = self.labels[inds].to(self.device)
        # split data in training and validation sets
        val_size = int(self.val_size * n)
        inds = torch.randperm(n)
        (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

        self.train["samples"] = self.samples[train_idx, :].to(self.device)
        self.train["labels"] = self.labels[train_idx].to(self.device)

        self.val["samples"] = self.samples[val_idx, :].to(self.device)
        self.val["labels"] = self.labels[val_idx].to(self.device)


def batch(data, labels, batch_size=1):
    n = data.shape[0]
    # mix data for train batches
    rand_perm = torch.randperm(n)
    data = data[rand_perm, :]
    labels = labels[rand_perm]

    batches = []
    for i in range(n // batch_size):
        inds = np.arange(i * batch_size, (i + 1) * batch_size, dtype=int)
        batches.append(
            (
                data[inds, :],
                labels[inds],
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
