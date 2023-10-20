#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.gan_mi.gan_cmi import GanCMI

DATA_PATH = "../data_cmi"
FILES = {
    "lf_10kdz20": {
        "data": "Linear_Data/catF/data.10k.dz20.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    },
    "nl_10kdz20": {
        "data": "Non_Linear_Data/catNon-lin-NI_6/data.10k.dz20.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_6/ksg_gt.dz20.npy",
    },
}


def read_data(key):
    data = np.load(f"{DATA_PATH}/{FILES[key]['data']}")
    true_cmi = np.load(f"{DATA_PATH}/{FILES[key]['ksg']}")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    return x, y, z, true_cmi[0]


def get_cmi(x, y, z, model_params, training_params):
    cmi = []
    for i in range(5):
        print(i + 1)
        cmigan_model = GanCMI(x, y, z, **model_params)
        cmigan_model.fit(**training_params, verbose=False)
        cmi.append(cmigan_model.get_cmi())
    return cmi


def cmiTest01():
    print("Test 01/02")
    # data
    data_keys = FILES.keys()

    # model parameters
    max_epochs = 5000
    batch_size = "full"
    noise_dim = 80

    # Training
    training_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-4,
        "lr_factor": 0.1,
        "lr_patience": 1000,
        "r_training_steps": 5,
        "g_training_steps": 1,
        "weight_decay": 1e-5,
    }
    results = []
    for key in data_keys:
        print(key)
        x, y, z, true_cmi = read_data(key)
        # Net parameters
        gdim = noise_dim + z.shape[1]
        model_params = {
            "noise_dim": noise_dim,
            "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
            "g_afn": "gelu",
            "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
            "r_afn": "gelu",
        }
        cmi = get_cmi(x, y, z, model_params, training_params)
        temp = [(key, true_cmi, "true value")]
        temp += [(key, x, "c-mi-gan") for x in cmi]
        temp = pd.DataFrame(temp, columns=["sim", "cmi", "method"])
        results.append(temp)
    results = pd.concat(results, ignore_index=True)
    sns.catplot(data=results, x="sim", y="cmi", hue="method", kind="bar")


def cmiTest02():
    print("Test 02/02")
    # data
    data_keys = FILES.keys()

    # model parameters
    max_epochs = 5000
    batch_size = "full"
    noise_dim = 80

    # Training
    training_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-4,
        "lr_factor": 0.1,
        "lr_patience": 1000,
        "r_training_steps": 5,
        "g_training_steps": 1,
        "stop_patience": 1000,
        "stop_min_delta": 0.00,
        "weight_decay": 1e-5,
    }
    for key in data_keys:
        print(key)
        x, y, z, true_cmi = read_data(key)
        # Net parameters
        gdim = noise_dim + z.shape[1]
        model_params = {
            "noise_dim": noise_dim,
            "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
            "g_afn": "gelu",
            "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
            "r_afn": "gelu",
        }

        cmigan_model = GanCMI(x, y, z, **model_params)
        cmigan_model.fit(**training_params, verbose=True)
        cmi = cmigan_model.get_cmi()
        (
            cmi_epoch,
            gen_loss_epoch,
            reg_loss_epoch,
            reg_loss_smooth_epoch,
        ) = cmigan_model.get_curves()

        fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
        epoch = np.arange(cmi_epoch.size)
        axs[0].set_title("C-MI-Gan")
        axs[0].plot(epoch, cmi_epoch, "r", label="val cmi")
        axs[0].axhline(true_cmi, color="b", label="true value")
        axs[0].axhline(cmi, color="k", label="estimated cmi")
        axs[0].legend(loc="lower right")
        axs[1].plot(epoch, gen_loss_epoch, "b", label="gen loss")
        axs[1].legend(loc="lower right")
        axs[2].plot(epoch, reg_loss_epoch, "r", label="reg loss")
        axs[2].plot(epoch, reg_loss_smooth_epoch, "k", label="smoothed reg loss")
        axs[2].set_xlabel("Epoch", fontweight="bold")
        axs[2].legend(loc="upper right")


if __name__ == "__main__":
    cmiTest01()
    cmiTest02()
    plt.show()
