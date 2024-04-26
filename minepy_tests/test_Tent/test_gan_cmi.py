#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.gan_mi.gan_cmi import GanCMI

DATA_PATH = "./data_cmigan/catNon-lin-NI_3/"
# DATA_PATH = "./data_cmigan"


def cmiLinear():
    data = np.load(f"{DATA_PATH}/data.20k.dz10.seed0.npy")
    cmi_teo = np.load(f"{DATA_PATH}/ksg_gt.dz10.npy")[0]

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    print(data.shape)

    # Net parameters
    max_epochs = 7000
    batch_size = "full"
    noise_dim = 80
    dz = z.shape[1]
    gdim = noise_dim + dz

    model_params = {
        "noise_dim": noise_dim,
        "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
        "g_afn": "gelu",
        "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
        "r_afn": "gelu",
    }
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
        "verbose": True,
    }

    class_cmigan_model = GanCMI(x, y, z, **model_params)
    class_cmigan_model.fit(**training_params)

    cmi = class_cmigan_model.get_cmi()
    (
        cmi_epoch,
        gen_loss_epoch,
        reg_loss_epoch,
        reg_marg_loss,
    ) = class_cmigan_model.get_curves()

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    epoch = np.arange(cmi_epoch.size)
    axs[0].set_title("")
    axs[0].plot(epoch, cmi_epoch, "r", label="")
    axs[0].axhline(cmi_teo, color="b", label="")
    axs[0].axhline(cmi, color="k", label="")
    axs[1].plot(epoch, gen_loss_epoch, "b", label="gen")
    axs[1].plot(epoch, reg_marg_loss, "r", label="gen")
    axs[2].plot(epoch, reg_loss_epoch, "r", label="reg")
    plt.show()


if __name__ == "__main__":
    cmiLinear()
