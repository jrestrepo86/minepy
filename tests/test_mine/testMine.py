#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.mine.mine import Mine
from tests.testTools import Progress, gausianSamples

NREA = 3  # number of realizations
MAX_ACTORS = 6  # number of nodes
N = 10000  # series data points
METHODS = ["mine_biased", "mine", "remine"]

# model parameters
general_model_params = {
    "hidden_layers": [64, 32, 16, 8, 4],
    "afn": "gelu",
}
# training parameters
training_params = {
    "batch_size": "full",
    "max_epochs": 20000,
    "lr": 1e-3,
    "lr_factor": 0.5,
    "lr_patience": 100,
    "stop_patience": 1000,
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
            "regWeight": 0.1,
            "targetVal": 0,
        }
    else:
        model_params = {
            **general_model_params,
        }
    return model_params


def testMine():
    print("MINE methods comparison")
    # gaussian noise parameters
    Rho = np.linspace(-0.98, 0.98, 11)
    Rho_label = [f"{x:.2f}" for x in Rho]
    sims_params = []
    results = []
    for i, rho in enumerate(Rho):
        # Generate data
        x, y, true_mi = gausianSamples(N, rho)
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
    results = pd.DataFrame(results, columns=["sim", "method", "mi"])
    sns.catplot(
        data=results,
        x="sim",
        y="mi",
        hue="method",
        kind="bar",
        hue_order=["true value", "mine", "remine", "mine_biased"],
    )


if __name__ == "__main__":
    testMine()
    plt.show()

# def plot(ax, Rho, teo_mi, mi, label):
#     ax.plot(Rho, teo_mi, ".k", label="True")
#     ax.plot(Rho, mi, "b", label=label)
#     ax.legend(loc="upper center")
#     ax.set_title(label)
#
#
# def testMine():
#     # Net
#     loss1 = "mine_biased"
#     loss2 = "mine"
#     loss3 = "remine"
#     model_params = {"hidden_dim": 50, "num_hidden_layers": 3, "afn": "elu"}
#
#     N = 3000
#     mu = np.array([0, 0])
#     Rho = np.linspace(-0.99, 0.99, 21)
#     mi_teo = np.zeros(*Rho.shape)
#     mi_mine_biased = np.zeros(*mi_teo.shape)
#     mi_mine = np.zeros(*mi_teo.shape)
#     mi_remine = np.zeros(*mi_teo.shape)
#
#     # Training
#     batch_size = "full"
#     max_epochs = 100000
#     train_params = {
#         "batch_size": batch_size,
#         "max_epochs": max_epochs,
#         "val_size": 0.2,
#         "lr": 1e-3,
#         "lr_factor": 0.5,
#         "lr_patience": 100,
#         "stop_patience": 300,
#         "stop_min_delta": 0.0,
#         "verbose": False,
#     }
#
#     for i, rho in enumerate(tqdm(Rho)):
#         # Generate data
#         cov_matrix = np.array([[1, rho], [rho, 1]])
#         joint_samples_train = np.random.multivariate_normal(
#             mean=mu, cov=cov_matrix, size=(N, 1)
#         )
#         X = np.squeeze(joint_samples_train[:, :, 0])
#         Y = np.squeeze(joint_samples_train[:, :, 1])
#
#         # Teoric value
#         mi_teo[i] = -0.5 * np.log(1 - rho**2)
#         # models
#         model_biased = Mine(X, Y, loss=loss1, **model_params)
#         model_mine = Mine(X, Y, loss=loss2, alpha=0.01, **model_params)
#         model_remine = Mine(
#             X, Y, loss=loss3, regWeight=0.1, targetVal=0, **model_params
#         )
#
#         # Train models
#         model_biased.fit(**train_params)
#         model_mine.fit(**train_params)
#         model_remine.fit(**train_params)
#
#         # Get mi estimation
#         mi_mine_biased[i] = model_biased.get_mi()
#         mi_mine[i] = model_mine.get_mi()
#         mi_remine[i] = model_remine.get_mi()
#
#     # Plot
#     fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
#     plot(axs[0], Rho, mi_teo, mi_mine_biased, label=loss1)
#     plot(axs[1], Rho, mi_teo, mi_mine, label=loss2)
#     plot(axs[2], Rho, mi_teo, mi_remine, label=loss3)
#     axs[0].set_xlabel("rho")
#     axs[0].set_ylabel("mi")
#     axs[1].set_xlabel("rho")
#     axs[2].set_xlabel("rho")
#     fig.suptitle("Mutual information neural estimation", fontsize=13)
#     plt.show()
