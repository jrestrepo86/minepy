#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt
from ray.experimental.tqdm_ray import tqdm

from minepy.class_mi.class_mi import ClassMI
from tests.testTools import Progress, gausianSamples

# from testTool import Progress, gaussianSamples


# from tqdm.auto import tqdm


NREA = 5  # number of realizations
MAX_ACTORS = 5  # number of nodes
N = 10000  # series data points
# Net parameters
model_params = {"hidden_layers": [32, 16, 8, 4], "afn": "gelu"}
# Training
training_params = {
    "batch_size": "full",
    "max_epochs": 3000,
    "lr": 1e-4,
    "weight_decay": 5e-5,
    "stop_patience": 600,
    "stop_min_delta": 0,
    "val_size": 0.2,
}


@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, rho, progress):
    model = ClassMI(x, y, **model_params)
    model.fit(**training_params, verbose=False)
    progress.update.remote()
    return (rho, model.get_mi())


def testClassMi01():
    print("Test 01/02")
    # gaussian noise parameter
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
    results += [(rho_key, "ccmi", mi) for rho_key, mi in res]
    results = pd.DataFrame(results, columns=["rho", "method", "mi"])
    sns.catplot(data=results, x="rho", y="mi", hue="method", kind="bar")


def testClassMi02():
    rho = 0.95
    # Generate data
    x, y, true_mi = gausianSamples(N, rho)
    # models
    class_mi_model = ClassMI(x, y, **model_params)

    class_mi_model.fit(**training_params, verbose=True)
    # Get mi estimation
    class_mi = class_mi_model.get_mi()
    val_mi, val_loss = class_mi_model.get_curves()

    print(f"true mi={true_mi}, estimated mi={class_mi}")
    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    axs[0].plot(val_mi, "r", label="val mi")
    axs[0].axhline(true_mi, color="k", label="true value")
    axs[0].axhline(class_mi, color="b", label="estimated mi")
    axs[0].set_title("Donsker-Varadhan representation")
    axs[0].legend(loc="lower right")

    axs[1].plot(val_loss, "r", label="Val")
    axs[1].set_title("Cross-Entropy loss")

    fig.suptitle(
        f"Curves for rho={rho},\n true mi={true_mi:.2f} and estim. mi={class_mi:.2f} ",
        fontsize=13,
    )


if __name__ == "__main__":
    testClassMi01()
    testClassMi02()
    plt.show()
