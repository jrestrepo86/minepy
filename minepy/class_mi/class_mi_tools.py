#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification mutual information tools
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

EPS = 1e-6


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


class class_mi_data_loader:
    def __init__(self, X, Y, val_size=0.2, device="cuda"):
        self.set_joint_marginals(X, Y)
        self.split_train_val(val_size, device)

    def set_joint_marginals(self, X, Y):
        n = X.shape[0]
        # set marginals
        X_marg = X[np.random.permutation(n), :]
        Y_marg = Y[np.random.permutation(n), :]
        data_joint = np.hstack((X, Y))
        data_marg = np.hstack((X_marg, Y_marg))
        samples = np.vstack((data_joint, data_marg))
        labels = np.squeeze(np.vstack((np.ones((n, 1)), np.zeros((n, 1)))))
        self.samples = samples
        self.labels = labels

        return samples, labels

    def split_train_val(self, val_size, device):
        n = self.samples.shape[0]
        # send data top device
        samples = torch.from_numpy(self.samples)
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
        self.samples = samples.to(device)
        self.labels = labels.to(device)
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

        # set joint  and marginal xyz
        inds = np.random.permutation(n)
        data_joint_xyz = np.hstack((X, Y, Z))
        data_marg_xyz = np.hstack((X, Y[inds, :], Z[inds, :]))
        self.samples_xyz = np.vstack((data_joint_xyz, data_marg_xyz))
        self.labels_xyz = np.squeeze(np.vstack((np.ones((n, 1)), np.zeros((n, 1)))))

        # set joint  and marginal xz
        data_joint_xz = np.hstack((X, Z))
        data_marg_xz = np.hstack((X, Z[np.random.permutation(n), :]))
        self.samples_xz = np.vstack((data_joint_xz, data_marg_xz))
        self.labels_xz = np.squeeze(np.vstack((np.ones((n, 1)), np.zeros((n, 1)))))

    def split_train_val(self, val_size, device):
        n = self.samples_xyz.shape[0]

        samples_xyz = torch.from_numpy(self.samples_xyz)
        labels_xyz = torch.from_numpy(self.labels_xyz)
        samples_xz = torch.from_numpy(self.samples_xz)
        labels_xz = torch.from_numpy(self.labels_xz)
        # mix samples
        inds = np.random.permutation(n)
        samples_xyz = samples_xyz[inds, :]
        labels_xyz = labels_xyz[inds]
        samples_xz = samples_xz[inds, :]
        labels_xz = labels_xz[inds]
        # split data in training and validation sets
        val_size = int(val_size * n)
        inds = torch.randperm(n)
        (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

        # send data to device
        self.samples_xyz = samples_xyz.to(device)
        self.labels_xyz = labels_xyz.to(device)
        self.samples_xz = samples_xz.to(device)
        self.labels_xz = labels_xz.to(device)

        self.train_samples_xyz = samples_xyz[train_idx, :].to(device)
        self.train_labels_xyz = labels_xyz[train_idx].to(device)
        self.train_samples_xz = samples_xz[train_idx, :].to(device)
        self.train_labels_xz = labels_xz[train_idx].to(device)

        self.val_samples_xyz = samples_xyz[val_idx, :].to(device)
        self.val_labels_xyz = labels_xyz[val_idx].to(device)
        self.val_samples_xz = samples_xz[val_idx, :].to(device)
        self.val_labels_xz = labels_xz[val_idx].to(device)


class class_cmi_gen_data_loader:
    def __init__(self, X, Y, Z, val_size=0.2, k=1, device="cuda"):
        self.set_joint_marginals(X, Y, Z, k)
        self.split_train_val(val_size, device)

    def set_joint_marginals(self, X, Y, Z, k):
        n = X.shape[0]

        # rand permute
        inds = torch.randperm(n)
        X, Y, Z = X[inds, :], Y[inds, :], Z[inds, :]
        # set joint
        # p(x,y,z)
        data_joint = np.hstack((X, Y, Z))

        # split data for the classifier and the generator
        m = np.ceil(n / k).astype(int)
        (class_idx, gen_idx) = (inds[:m], inds[m:])
        Z_class, Z_gen = Z[class_idx, :], Z[gen_idx, :]
        Y_gen = Y[gen_idx, :]

        # set marginals - Gerate y|z samples (knn)
        # algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’},
        # nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(Z_gen)
        nbrs = NearestNeighbors(
            n_neighbors=k,
            algorithm="kd_tree",
            metric="euclidean",
        ).fit(Z_gen)
        nn_indx = nbrs.kneighbors(Z_class, return_distance=False).flatten()
        class_idx = torch.repeat_interleave(class_idx, k)
        # p(x,z)p(y|z)
        data_marg = np.hstack((X[class_idx, :], Y_gen[nn_indx, :], Z[class_idx, :]))

        self.samples = np.vstack((data_joint, data_marg))
        self.labels = np.squeeze(
            np.vstack(
                (np.ones((data_joint.shape[0], 1)), np.zeros((data_marg.shape[0], 1)))
            )
        )

    def split_train_val(self, val_size, device):
        n = self.samples.shape[0]
        # send data top device
        samples = torch.from_numpy(self.samples)
        labels = torch.from_numpy(self.labels)

        # split data in training and validation sets
        val_size = int(val_size * n)
        inds = torch.randperm(n)
        (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

        self.samples = samples.to(device)
        self.labels = labels.to(device)

        self.train_samples = samples[train_idx, :].to(device)
        self.train_labels = labels[train_idx].to(device)

        self.val_samples = samples[val_idx, :].to(device)
        self.val_labels = labels[val_idx].to(device)
