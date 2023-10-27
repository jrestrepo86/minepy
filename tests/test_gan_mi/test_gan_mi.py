#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.gan_mi.gan_mi import GanMI
from tests.testTools import Progress, gausianSamples

NREA = 5  # number of realizations
MAX_ACTORS = 5  # number of nodes
N = 10000  # series data points
# Net parameters
noise_dim = 16
gdim = noise_dim
model_params = {
    "noise_dim": noise_dim,
    "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
    "g_afn": "gelu",
    "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
    "r_afn": "gelu",
}
# Training
training_params = {
    "batch_size": "full",
    "max_epochs": 5000,
    "lr": 1e-4,
    "lr_factor": 0.1,
    "lr_patience": 1000,
    "r_training_steps": 5,
    "g_training_steps": 1,
    "weight_decay": 1e-5,
}


# @ray.remote(num_cpus= MAX_ACTORS)
@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, rho, progress):
    model = GanMI(x, y, **model_params)
    model.fit(**training_params, verbose=False)
    progress.update.remote()
    return (rho, model.get_mi())


def testGanMi01():
    print("Test 01/02")
    # gaussian noise parameters
    Rho = np.linspace(-0.98, 0.98, 21)
    Rho_label = [f"{x:.2f}" for x in Rho]
    sims_params = []
    results = []

    for i, rho in enumerate(Rho):
        # Generate data
        x, y, true_mi = gausianSamples(N, rho)
        results += [(Rho_label[i], "true value", true_mi)]
        x_id = ray.put(x)
        y_id = ray.put(y)
        for _ in range(NREA):
            sim_params_ = {
                "x": x_id,
                "y": y_id,
                "rho": Rho_label[i],
            }
            sims_params.append(sim_params_)

    progress = Progress.remote(len(sims_params))
    random.shuffle(sims_params)
    res = ray.get(
        [
            model_training.remote(
                s["x"],
                s["y"],
                s["rho"],
                progress,
            )
            for s in sims_params
        ]
    )
    # parse-results
    results += [(rho_key, "c-mi-gan", mi) for rho_key, mi in res]
    results = pd.DataFrame(results, columns=["rho", "method", "mi"])
    sns.catplot(data=results, x="rho", y="mi", hue="method", kind="bar")


def testGanMi02():
    rho = 0.95
    # Generate data
    x, y, true_mi = gausianSamples(N, rho)
    # models
    gan_mi_model = GanMI(x, y, **model_params)

    gan_mi_model.fit(**training_params, verbose=True)
    # Get mi estimation
    gan_mi = gan_mi_model.get_mi()
    Dkl_val, gen_loss, reg_loss = gan_mi_model.get_curves()

    print(f"true mi={true_mi}, estimated mi={gan_mi}")
    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    axs[0].plot(Dkl_val, "r", label="Val")
    axs[0].set_title("Donsker-Varadhan representation")
    axs[0].legend(loc="lower right")
    axs[1].plot(gen_loss, "r")
    axs[1].set_title("Generator loss")
    axs[2].plot(reg_loss, "r")
    axs[2].set_title("Regresor loss")

    fig.suptitle(
        f"Curves for rho={rho}, true mi={true_mi:.2f} and estim. mi={gan_mi:.2f} ",
        fontsize=13,
    )


if __name__ == "__main__":
    testGanMi01()
    # testGanMi02()
    plt.show()
