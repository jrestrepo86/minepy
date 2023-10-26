#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification mutual information tools
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class class_mi_data_loader:
    def __init__(self, X, Y, val_size=0.2, device="cuda"):
        self.set_joint_marginals(X, Y)
        self.split_train_val(val_size, device)

    def set_joint_marginals(self, X, Y):
        n = X.shape[0]
        # set marginals
        Xmarg = X[np.random.permutation(n), :]
        Ymarg = Y[np.random.permutation(n), :]
        data_joint = np.hstack((X, Y))
        data_marg = np.hstack((Xmarg, Ymarg))
        samples = np.vstack((data_joint, data_marg))
        joint_labels = np.hstack((np.ones((n, 1)), np.zeros((n, 1))))
        marg_labels = np.hstack((np.zeros((n, 1)), np.ones((n, 1))))
        labels = np.squeeze(np.vstack((joint_labels, marg_labels)))
        self.samples = samples
        self.labels = labels

        return samples, labels

    def split_train_val(self, val_size, device):
        n = self.samples.shape[0]
        # send data top device
        samples = torch.from_numpy(self.samples.astype(np.float32))
        labels = torch.from_numpy(self.labels)

        # mix samples
        inds = np.random.permutation(n)
        samples = samples[inds, :]
        labels = labels[inds]
        # split data in training and validation sets
        val_size = int(val_size * n)
        inds = torch.randperm(n)
        (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

        train_samples = samples[train_idx, :]
        train_labels = labels[train_idx]

        val_samples = samples[val_idx, :]
        val_labels = labels[val_idx]
        self.train_samples = train_samples.to(device)
        self.train_labels = train_labels.to(device)
        self.val_samples = val_samples.to(device)
        self.val_labels = val_labels.to(device)


class class_cmi_diff_data_loader:
    def __init__(self, X, Y, Z, val_size=0.2, device="cuda"):
        self.set_joint_marginals(X, Y, Z)
        self.split_train_val(val_size, device)

    def set_joint_marginals(self, X, Y, Z):
        n = X.shape[0]

        # set joint xyz
        data_joint_xyz = np.hstack((X, Y, Z))
        # set marginal xyz
        inds1 = np.random.permutation(n)
        inds2 = np.random.permutation(n)
        data_marg_xyz = np.hstack((X[inds1, :], Y[inds2, :], Z[inds2, :]))
        joint_labels = np.hstack((np.ones((n, 1)), np.zeros((n, 1))))
        marg_labels = np.hstack((np.zeros((n, 1)), np.ones((n, 1))))
        self.samples_xyz = np.vstack((data_joint_xyz, data_marg_xyz))
        self.labels_xyz = np.squeeze(np.vstack((joint_labels, marg_labels)))

        # set joint xz
        data_joint_xz = np.hstack((X, Z))
        # set marginal xz
        inds1 = np.random.permutation(n)
        inds2 = np.random.permutation(n)
        data_marg_xz = np.hstack((X[inds1, :], Z[inds2, :]))
        self.samples_xz = np.vstack((data_joint_xz, data_marg_xz))
        self.labels_xz = np.squeeze(np.vstack((joint_labels, marg_labels)))

    def split_train_val(self, val_size, device):
        n = self.samples_xyz.shape[0]

        samples_xyz = torch.from_numpy(self.samples_xyz.astype(np.float32))
        labels_xyz = torch.from_numpy(self.labels_xyz)
        samples_xz = torch.from_numpy(self.samples_xz.astype(np.float32))
        labels_xz = torch.from_numpy(self.labels_xz)

        # split data in training and validation sets
        inds = np.random.permutation(n)
        val_size = int(val_size * n)
        inds = torch.randperm(n)
        (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

        # send data to device
        self.train_samples_xyz = samples_xyz[train_idx, :].to(device)
        self.train_labels_xyz = labels_xyz[train_idx].to(device)
        self.train_samples_xz = samples_xz[train_idx, :].to(device)
        self.train_labels_xz = labels_xz[train_idx].to(device)

        self.val_samples_xyz = samples_xyz[val_idx, :].to(device)
        self.val_labels_xyz = labels_xyz[val_idx].to(device)
        self.val_samples_xz = samples_xz[val_idx, :].to(device)
        self.val_labels_xz = labels_xz[val_idx].to(device)


class class_cmi_gen_data_loader:
    def __init__(self, X, Y, Z, val_size=0.2, device="cuda"):
        self.set_joint_marginals(X, Y, Z)
        self.split_train_val(val_size, device)

    def knn_search(self, Zclass, Zgen):
        # knn
        nbrs = NearestNeighbors(
            n_neighbors=1,
            algorithm="kd_tree",  # algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
            metric="euclidean",
        ).fit(Zgen)
        nn_idx = nbrs.kneighbors(Zclass, return_distance=False).flatten()
        return nn_idx

    def set_joint_marginals(self, X, Y, Z):
        # set joint p(x,y,z)
        data_joint = np.hstack((X, Y, Z))
        # set marginals - Gerate y|z samples (knn)
        # split data for the classifier and the generator (knn)
        n = X.shape[0]
        rand_perm = np.random.permutation(n)
        data_marg = np.array([]).reshape(0, X.shape[1] + Y.shape[1] + Z.shape[1])
        for inds in np.split(rand_perm, 2, axis=0):
            mask = np.ones(n, dtype=bool)
            mask[inds] = False
            class_idx, gen_idx = mask, np.logical_not(mask)
            Ygen = Y[class_idx, :]
            Zclass, Zgen = Z[class_idx, :], Z[gen_idx, :]
            nn_idx = self.knn_search(Zclass, Zgen)
            # p(x,z)p(y|z)
            temp = np.hstack((X[class_idx, :], Ygen[nn_idx, :], Z[class_idx, :]))
            data_marg = np.vstack((data_marg, temp))

        # set samples
        samples = np.vstack((data_joint, data_marg))
        # set labels
        joint_labels = np.hstack((np.ones((n, 1)), np.zeros((n, 1))))
        marg_labels = np.hstack((np.zeros((n, 1)), np.ones((n, 1))))
        labels = np.squeeze(np.vstack((joint_labels, marg_labels)))
        self.samples = samples
        self.labels = labels

    def split_train_val(self, val_size, device):
        n = self.samples.shape[0]
        samples = torch.from_numpy(self.samples.astype(np.float32))
        labels = torch.from_numpy(self.labels)

        # split data in training and validation sets
        inds = torch.randperm(n)
        val_size = int(val_size * n)
        (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

        # send data to device
        self.train_samples = samples[train_idx, :].to(device)
        self.train_labels = labels[train_idx].to(device)

        self.val_samples = samples[val_idx, :].to(device)
        self.val_labels = labels[val_idx].to(device)
