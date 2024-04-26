# -*- coding: utf-8 -*-

import os
from itertools import product

import numpy as np
import ray
from numpy.linalg import det
from ray.experimental.tqdm_ray import tqdm

EPS = 1e-10

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


def get_z(N, dz, rng):
    choice = rng.randint(2)
    if choice == 0:
        return rng.laplace(size=(N, dz)) * 0.01
    elif choice == 1:
        return rng.uniform(-1, 1, size=(N, dz)) * 0.01
    else:
        return rng.randn(N, dz) * 0.01


def non_lin_fun(x, rng):
    choice = rng.randint(5)
    # choice = rng.randint(6)
    # print(choice)
    x = x / x.std()

    if choice == 0:
        return x**3
    elif choice == 1:
        return 1 / (x - x.min() + 1)
    elif choice == 2:
        return np.exp(-x)
    elif choice == 3:
        return np.log1p(x - np.min(x))
    elif choice == 4:
        return 1 / (1 + np.exp(-x))
    elif choice == 5:
        return np.cos(x)
    else:
        return np.exp(-np.abs(x))


def gaussian_samples(N, rho=None, random_state=None):
    rng = np.random.RandomState(random_state)
    if rho is None:
        rho = rng.uniform(-1 + EPS, 1 - EPS)

    mu = np.array([0, 0])
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples = rng.multivariate_normal(mean=mu, cov=cov_matrix, size=(N, 1))
    x = joint_samples[:, :, 0]
    y = joint_samples[:, :, 1]

    est_mi = -0.5 * np.log(det(cov_matrix))
    true_mi = -0.5 * np.log(1 - rho**2)
    return x, y, true_mi, est_mi


def read_data(key):
    """
    Data obtained from:
    https://github.com/arnabkmondal/C-MI-GAN
    """
    data = np.load(f"{DATA_PATH}/{FILES[key]['data']}")
    true_cmi = np.load(f"{DATA_PATH}/{FILES[key]['ksg']}")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    return x, y, z, true_cmi[0]


def cmi_non_lin_samples01(N, dz, rho=None, random_state=None):
    """
    https://github.com/baosws/DINE/blob/main/src/data/data_gen.py
    """
    rng = np.random.RandomState(random_state)
    if rho is None:
        rho = rng.uniform(-1 + EPS, 1 - EPS)

    xp, yp, true_cmi, est_cmi = gaussian_samples(N, rho=rho, random_state=random_state)
    z = get_z(N, dz, rng)
    a, b = rng.randn(dz, 1), rng.randn(dz, 1)
    x = non_lin_fun(z @ a + xp, rng)
    y = non_lin_fun(z @ b + yp, rng)
    return x, y, z, true_cmi, est_cmi


def cmi_non_lin_samples02(N, dz, random_state=None):
    """
    https://github.com/arnabkmondal/C-MI-GAN/
    """
    rng = np.random.RandomState(random_state)
    n1, n2, _, _ = gaussian_samples(N, rho=0, random_state=random_state)
    n1, n2 = n1 * 0.1, n2 * 0.1
    z = rng.randn(N, dz) + 1
    a = rng.randn(dz, 1)
    a = a / np.sqrt((a**2).sum())
    x = non_lin_fun(n1, rng)
    y = non_lin_fun(z @ a + 2 * n1 + n2, rng)
    return x, y, z


@ray.remote
class Progress:
    def __init__(self, max_it=1, pbar=True):
        self.pbar_flag = pbar
        self.pbar = tqdm(total=max_it)
        self.count = -1
        self.max_it = max_it
        self.update()

    def update(self):
        if self.pbar_flag:
            self.pbar.update()
        else:
            self.count += 1
            p = 100 * self.count / self.max_it
            print(f"Progress: {p:.2f}%  {self.count}/{self.max_it}")
