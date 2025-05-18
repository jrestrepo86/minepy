#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from typing import Counter

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.class_mi.class_mi import ClassMI
from minepy.gan_mi.gan_mi import GanMI
from minepy.mine.mine import Mine
from minepy_tests.testTools import Progress, gaussianSamples

NREA = 6  # number of realizations
MAX_ACTORS = 7  # number of nodes
N = 3000  # series data points
METHODS = ["mine_biased", "mine", "remine", "cmi-gen", "mi-gan"]


# model parameters
general_model_params = {
    "hidden_layers": [32, 16, 8, 4],
    "afn": "gelu",
}
# training parameters
general_training_params = {
    "batch_size": "full",
    "max_epochs": 40000,
    "lr": 1e-6,
    "weight_decay": 5e-5,
    "stop_patience": 1000,
    "stop_min_delta": 0.0,
}


def set_model_parameters(method):
    if method == "mine":
        model_params = {
            **general_model_params,
            "loss": method,
            "alpha": 0.01,
        }
        training_params = {
            **general_training_params,
            "val_size": 0.2,
        }
    elif method == "remine":
        model_params = {
            **general_model_params,
            "loss": method,
            "regWeight": 0.1,
            "targetVal": 0,
        }
        training_params = {
            **general_training_params,
            "val_size": 0.2,
        }
    elif method == "mine_biased":
        model_params = {
            **general_model_params,
            "loss": method,
        }
        training_params = {
            **general_training_params,
            "val_size": 0.2,
        }
    elif method == "cmi-gen":
        model_params = {
            **general_model_params,
        }
        training_params = {
            **general_training_params,
            "val_size": 0.2,
        }
    else:
        noise_dim = 40
        gdim = noise_dim + 1
        model_params = {
            "noise_dim": noise_dim,
            "g_hidden_layers": [gdim, gdim / 2],
            "g_afn": "gelu",
            "r_hidden_layers": [gdim / 2, gdim / 4],
            "r_afn": "gelu",
        }
        training_params = {
            **general_training_params,
            "r_training_steps": 5,
            "g_training_steps": 1,
        }

    return model_params, training_params


@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, sim, method, model_params, training_params, progress):
    if method in ["mine", "remine", "mine_biased"]:
        model = Mine(x, y, **model_params)
    elif method == "cmi-gen":
        model = ClassMI(x, y, **model_params)
    else:
        model = GanMI(x, y, **model_params)
    model.fit(**training_params, verbose=False)
    progress.update.remote()
    return (sim, method, model.get_mi())


def miTest():
    print("MINE methods comparison")
    # gaussian noise parameters
    Rho = np.linspace(-0.98, 0.98, 11)
    Rho_label = [f"{x:.2f}" for x in Rho]
    sims_params = []
    results = []
    for i, rho in enumerate(Rho):
        # Generate data
        x, y, true_mi = gaussianSamples(N, rho)
        results += [(Rho_label[i], "true value", true_mi)]
        x_id = ray.put(x)
        y_id = ray.put(y)
        for method in METHODS:
            model_params, training_params = set_model_parameters(method)
            for _ in range(NREA):
                sim_params_ = {
                    "x": x_id,
                    "y": y_id,
                    "rho": Rho_label[i],
                    "method": method,
                    "model_params": model_params,
                    "training_params": training_params,
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
                s["training_params"],
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
        hue_order=["true value"] + METHODS,
    )


if __name__ == "__main__":
    miTest()
    plt.show()
