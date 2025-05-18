#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.class_mi.class_diff_cmi import ClassDiffCMI
from minepy_tests.testTools import Progress, read_data

FILES = ["lf_10kdz20", "nl_10kdz20"]

NREA = 5  # number of realizations
MAX_ACTORS = 5  # number of nodes

# model parameters
afn = "gelu"

# Training
training_params = {
    "batch_size": "full",
    "max_epochs": 8000,
    "lr": 1e-6,
    "weight_decay": 1e-5,
    "stop_patience": 1000,
    "stop_min_delta": 0.00,
}


# @ray.remote(num_cpus= MAX_ACTORS)
@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, z, model_params, sim, progress):
    model = ClassDiffCMI(x, y, z, **model_params)
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
        gdim = 2 + z.shape[1]
        for _ in range(NREA):
            sim_params_ = {
                "x": x_id,
                "y": y_id,
                "z": z_id,
                "sim": sim,
                "model_params": {
                    "hidden_layers_xyz": [gdim / 2, gdim / 4, gdim / 8],
                    "hidden_layers_xz": [gdim / 2, gdim / 4, gdim / 8],
                    "afn": afn,
                },
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
    results += [(sim_key, "ccmi-diff", cmi) for sim_key, cmi in res]
    results = pd.DataFrame(results, columns=["sim", "method", "cmi"])
    # plot
    sns.catplot(data=results, x="sim", y="cmi", hue="method", kind="bar")


def cmiTest02():
    print("Test 02/02")
    sims = FILES

    for sim in sims:
        print(sim)
        # data
        x, y, z, true_cmi = read_data(sim)
        # model parameters
        gdim = 2 + z.shape[1]
        model_params = {
            "hidden_layers_xyz": [gdim / 2, gdim / 4, gdim / 8],
            "hidden_layers_xz": [gdim / 2, gdim / 4, gdim / 8],
            "afn": afn,
        }

        class_cmigen_model = ClassDiffCMI(x, y, z, **model_params)
        class_cmigen_model.fit(**training_params, verbose=True)
        cmi = class_cmigen_model.get_cmi()
        (
            val_cmi_epoch,
            val_dkl_epoch_xyz,
            val_loss_epoch_xyz,
            val_dkl_epoch_xz,
            val_loss_epoch_xz,
        ) = class_cmigen_model.get_curves()

        fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
        epoch = np.arange(val_cmi_epoch.size)
        axs[0].set_title(f"CCMI-Diff sim {sim}")
        axs[0].plot(epoch, val_cmi_epoch, "b", label="val cmi")
        axs[0].axhline(true_cmi, color="g", label="true value")
        axs[0].axhline(cmi, color="k", label="estimated cmi")
        axs[0].legend(loc="lower right")
        axs[1].plot(epoch, val_dkl_epoch_xyz, "r", label="val dkl xyz")
        axs[1].plot(epoch, val_dkl_epoch_xz, "b", label="val dkl xy")
        axs[1].legend(loc="upper right")
        axs[2].plot(epoch, val_loss_epoch_xyz, "r", label="val loss xyz")
        axs[2].plot(epoch, val_loss_epoch_xz, "b", label="val loss xy")
        axs[2].legend(loc="upper right")
        axs[2].set_xlabel("Epoch", fontweight="bold")


if __name__ == "__main__":
    cmiTest01()
    cmiTest02()
    plt.show()
