#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.mine.mine import Mine
from minepy_tests.testTools import Progress, gaussian_samples

NREA = 6  # number of realizations
MAX_ACTORS = 12  # number of nodes (parallel computing)
N = 3000  # series data points
METHODS = ["mine_biased", "mine", "remine"]


# model parameters
general_model_params = {
    "hidden_layers": [4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4],
    "afn": "relu",
}
# training parameters
training_params = {
    "batch_size": "full",
    "max_epochs": 40000,
    "lr": 1e-3,
    "weight_decay": 5e-5,
    "stop_patience": 500,
    "stop_min_delta": 0.0,
    "val_size": 0.2,
}


@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, sim, method, model_params, progress):
    model = Mine(x, y, loss=method, **model_params)
    model.fit(**training_params, verbose=False)
    progress.update.remote()
    return (sim, method, model.get_mi())


def set_model_parameters(method):
    if method == "mine":
        model_params = {
            **general_model_params,
            "alpha": 0.01,
        }
    elif method == "remine":
        model_params = {
            **general_model_params,
            "regWeight": 0.5,
            "targetVal": 0,
        }
    else:
        model_params = {
            **general_model_params,
        }
    return model_params


def testMine01():
    print("MINE methods comparison")
    # gaussian noise parameters
    Rho = np.linspace(-0.98, 0.98, 11)
    Rho_label = [f"{x:.2f}" for x in Rho]
    sims_params = []
    results = []
    for i, rho in enumerate(Rho):
        # Generate data
        x, y, true_mi, _ = gaussian_samples(N, rho)
        results += [(Rho_label[i], "true value", true_mi)]
        x_id = ray.put(x)
        y_id = ray.put(y)
        for method in METHODS:
            model_params = set_model_parameters(method)
            for _ in range(NREA):
                sim_params_ = {
                    "x": x_id,
                    "y": y_id,
                    "rho": Rho_label[i],
                    "method": method,
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
                s["rho"],
                s["method"],
                s["model_params"],
                progress,
            )
            for s in sims_params
        ]
    )
    # parse results
    results += [(sim_key, method, mi) for sim_key, method, mi in res]
    results = pd.DataFrame(results, columns=["rho", "method", "mi"])
    sns.catplot(
        data=results,
        x="rho",
        y="mi",
        hue="method",
        kind="bar",
        hue_order=["true value", "mine", "remine", "mine_biased"],
    )


def testMine02():
    for method in METHODS:
        rho = 0.58
        # Generate data
        x, y, true_mi, _ = gaussian_samples(N, rho)
        # model parameters
        model_params = set_model_parameters(method)
        # models
        model = Mine(x, y, **model_params)

        model.fit(**training_params, verbose=True)
        # Get mi estimation
        mi = model.get_mi()

        (
            val_loss_epoch,
            val_ema_loss_epoch,
            val_mi_epoch,
            test_mi_epoch,
        ) = model.get_curves()

        print(f"Method: {method} true mi={true_mi}, estimated mi={mi}")
        # Plot
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
        axs[0].plot(val_mi_epoch, "r", label="val mi")
        axs[0].plot(test_mi_epoch, "b", label="test mi")
        axs[0].axhline(true_mi, color="k", label="true value")
        axs[0].axhline(mi, color="g", label="estimated mi")
        axs[0].set_title(f"{method}")
        axs[0].legend(loc="lower right")

        axs[1].plot(val_loss_epoch, "r", label="val loss")
        axs[1].plot(val_ema_loss_epoch, "b", label="val loss smoothed")
        axs[1].legend(loc="upper right")

        fig.suptitle(
            f"Curves for rho={rho},\n true mi={true_mi:.2f} and estim. mi={mi:.2f} ",
            fontsize=13,
        )


if __name__ == "__main__":
    testMine01()
    testMine02()
    plt.show()
