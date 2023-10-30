# -*- coding: utf-8 -*-

import os

import numpy as np
import ray
from ray.experimental.tqdm_ray import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = f"{ROOT_DIR}/data_cmi"
FILES = {
    "lf_20kdz10": {
        "data": "Linear_Data/catF/data.20k.dz10.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz10.npy",
    },
    "lf_5kdz20": {
        "data": "Linear_Data/catF/data.5k.dz20.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    },
    "lf_10kdz20": {
        "data": "Linear_Data/catF/data.10k.dz20.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    },
    "lf_20kdz100": {
        "data": "Linear_Data/catF/data.20k.dz100.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz100.npy",
    },
    "nl_5kdz10": {
        "data": "Non_Linear_Data/catNon-lin-NI_1/data.5k.dz10.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_1/ksg_gt.dz10.npy",
    },
    "nl_10kdz10": {
        "data": "Non_Linear_Data/catNon-lin-NI_2/data.10k.dz10.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_2/ksg_gt.dz10.npy",
    },
    "nl_5kdz20": {
        "data": "Non_Linear_Data/catNon-lin-NI_5/data.5k.dz20.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_5/ksg_gt.dz20.npy",
    },
    "nl_10kdz20": {
        "data": "Non_Linear_Data/catNon-lin-NI_6/data.10k.dz20.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_6/ksg_gt.dz20.npy",
    },
    "nl_5kdz100": {
        "data": "Non_Linear_Data/catNon-lin-NI_13/data.5k.dz100.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_13/ksg_gt.dz100.npy",
    },
    "nl_10kdz100": {
        "data": "Non_Linear_Data/catNon-lin-NI_14/data.10k.dz100.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_14/ksg_gt.dz100.npy",
    },
}


@ray.remote
class Progress:
    def __init__(self, max_it=1):
        self.pbar = tqdm(total=max_it)
        self.count = 0

    def update(self):
        self.pbar.update()


def gaussianSamples(n, rho):
    mu = np.array([0, 0])
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples_train = np.random.multivariate_normal(
        mean=mu, cov=cov_matrix, size=(n, 1)
    )
    x = np.squeeze(joint_samples_train[:, :, 0])
    y = np.squeeze(joint_samples_train[:, :, 1])

    true_mi = -0.5 * np.log(1 - rho**2)
    return x, y, true_mi


def read_data(key):
    data = np.load(f"{DATA_PATH}/{FILES[key]['data']}")
    true_cmi = np.load(f"{DATA_PATH}/{FILES[key]['ksg']}")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    return x, y, z, true_cmi[0]
