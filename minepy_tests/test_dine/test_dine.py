#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.dine.dine_cmi import DineCMI
from minepy.gan_mi.gan_cmi import GanCMI
from minepy_tests.testTools import (FILES, Progress, cmi_non_lin_samples01,
                                    cmi_non_lin_samples02, read_data)

FILES = [
    "lf_5kdz20",
    "lf_10kdz20",
    "nl_5kdz10",
    "nl_10kdz20",
]

# FILE = list(FILES.keys())

NREA = 5  # number of realizations
MAX_ACTORS = 5  # number of nodes


gan_training_params = {
    "batch_size": "full",
    "max_epochs": 20000,
    "g_base_lr": 1e-8,
    "g_max_lr": 5e-5,
    "r_base_lr": 1e-8,
    "r_max_lr": 1e-4,
    "weight_decay": 1e-5,
    "stop_patience": 1000,
    "stop_min_delta": 0.00,
    "r_training_steps": 5,
    "g_training_steps": 1,
}
# model parameters
model_params = {"n_components": 16, "hidden_sizes": [4], "afn": "relu"}

training_params = {
    # "batch_size": "full",
    "batch_size": 64,
    "max_epochs": 500,
    "lr": 1e-8,
    "weight_decay": 5e-5,
}


# @ray.remote(num_cpus= MAX_ACTORS)
@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, z, model_params, sim, progress):
    model = DineCMI(x, y, z, **model_params)
    model.fit(**training_params)
    progress.update.remote()
    return (sim, model.get_cmi())


def cmiTest01():
    print("Test 01/02")
    sims = FILES
    results = []
    sims_params = []
    # Simulations
    for sim in sims:
        # data
        x, y, z, true_cmi = read_data(sim)
        results += [(sim, "true value", true_cmi)]
        x_id = ray.put(x)
        y_id = ray.put(y)
        z_id = ray.put(z)
        for _ in range(NREA):
            sim_params_ = {
                "x": x_id,
                "y": y_id,
                "z": z_id,
                "sim": sim,
                "model_params": model_params,
            }
            sims_params.append(sim_params_)

    progress = Progress.remote(len(sims_params))
    random.shuffle(sims_params)
    res = ray.get(
        [
            model_training.remote(
                s["x"],
                s["y"],
                s["z"],
                s["model_params"],
                s["sim"],
                progress,
            )
            for s in sims_params
        ]
    )
    # parse-results
    results += [(sim_key, "dine", cmi) for sim_key, cmi in res]
    results = pd.DataFrame(results, columns=["sim", "method", "cmi"])
    # plot
    sns.catplot(data=results, x="sim", y="cmi", hue="method", kind="bar")


def cmiTest02():
    print("Test 02/02")
    N = 50000
    sims = FILES
    RHO = [0.9]
    # Simulations
    # for sim in sims:
    for rho in RHO:
        # data
        # x, y, z, true_cmi = read_data(sim)
        x, y, z, true_cmi, est_cmi = cmi_non_lin_samples01(
            N, 4, rho=rho, random_state=3
        )
        # x, y, z = cmi_non_lin_samples02(N, 4, random_state=2)
        # true_cmi = 0

        dine_model = DineCMI(x, y, z, **model_params)
        dine_model.fit(**training_params, verbose=True)
        cmi = dine_model.get_cmi()
        val_loss_epoch, train_loss_epoch, cmi_epoch = dine_model.get_curves()

        gdim = 16 + z.shape[1]
        gan_model_params = {
            "noise_dim": 16,
            "g_hidden_layers": [gdim / 2, gdim / 4],
            "g_afn": "gelu",
            "r_hidden_layers": [gdim / 4, gdim / 8],
            "r_afn": "gelu",
        }

        cmigan_model = GanCMI(x, y, z, **gan_model_params)
        cmigan_model.fit(**gan_training_params, verbose=True)
        gan_cmi = cmigan_model.get_cmi()

        print(f"true={true_cmi}, estimated={cmi}, gan_cmi={gan_cmi}")

        fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
        epoch = np.arange(val_loss_epoch.size)
        axs[0].set_title(f"Dine - rho: {rho}")
        axs[0].plot(epoch, cmi_epoch, "r", label="cmi epoch")
        axs[0].axhline(true_cmi, color="b", label="true value")
        # axs[0].axhline(est_cmi, color="k", label="data value")
        axs[0].axhline(cmi, color="g", label="cmi")
        axs[0].legend(loc="lower right")
        axs[1].plot(epoch, train_loss_epoch, "b", label="train loss")
        axs[1].plot(epoch, val_loss_epoch, "r", label="val loss")
        # axs[1].legend(loc="lower right")
        # axs[2].plot(epoch, reg_loss_epoch, "r", label="reg loss")
        # axs[2].plot(epoch, reg_loss_smooth_epoch, "k", label="smoothed reg loss")
        # axs[2].set_xlabel("Epoch", fontweight="bold")
        # axs[2].legend(loc="upper right")


if __name__ == "__main__":
    # cmiTest01()
    cmiTest02()
    plt.show()
