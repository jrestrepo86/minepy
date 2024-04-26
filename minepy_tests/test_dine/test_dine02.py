#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.dine.DINE import DINE
from minepy.dine.dine_cmi import DineCMI
from minepy_tests.test_dine.test_dine import cmiTest02, model_training
from minepy_tests.testTools import Progress, cmi_non_lin_samples01

NREA = 5  # number of realizations
MAX_ACTORS = 7  # number of nodes


N = 3000  # series data points
DZ = 4
METHODS = ["DINE", "dineAlg01"]

training_paramsDINE = {
    "n_components": 8,
    "hidden_sizes": 4,
    "lr": 5e-3,
    "weight_decay": 5e-5,
    "max_epochs": 100,
    "random_state": None,
    "gpus": 0,
    "return_latents": False,
    "verbose": False,
}
model_paramsDineCMI = {
    "n_components": 8,
    "hidden_sizes": 4,
    "afn": "relu",
    "norm": True,
}
training_paramsDineCMI = {
    "batch_size": 64,
    "val_size": 0.3,
    "lr": 5e-3,
    "weight_decay": 5e-5,
    "max_epochs": 100,
    "verbose": False,
}
random_state = 4


@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, z, rho, method, progress):
    if method == "DINE":
        cmi = DINE(x, y, z, **training_paramsDINE)
    else:
        model = DineCMI(x, y, z, **model_paramsDineCMI)
        model.fit(**training_paramsDineCMI)
        cmi = model.get_cmi()
    progress.update.remote()
    return (rho, cmi, method)


def testDINE01():
    print("Test 01/02")
    # gaussian noise parameters
    Rho = np.linspace(-0.98, 0.98, 11)
    Rho_label = [f"{x:.2f}" for x in Rho]
    sims_params = []
    results = []

    for method in METHODS:
        for i, rho in enumerate(Rho):
            # Generate data

            x, y, z, true_cmi, _ = cmi_non_lin_samples01(
                N, DZ, rho=rho, random_state=random_state
            )

            results += [(Rho_label[i], "true value", true_cmi)]
            x_id = ray.put(x)
            y_id = ray.put(y)
            z_id = ray.put(z)
            for _ in range(NREA):
                sim_params_ = {
                    "x": x_id,
                    "y": y_id,
                    "z": z_id,
                    "method": method,
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
                s["z"],
                s["rho"],
                s["method"],
                progress,
            )
            for s in sims_params
        ]
    )
    # parse-results
    results += [(rho_key, method, cmi) for rho_key, cmi, method in res]
    results = pd.DataFrame(results, columns=["rho", "method", "cmi"])
    sns.catplot(data=results, x="rho", y="cmi", hue="method", kind="bar")


if __name__ == "__main__":
    testDINE01()
    plt.show()
