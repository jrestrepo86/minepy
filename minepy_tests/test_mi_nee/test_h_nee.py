#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.mi_nee.h_nee import HNee
from minepy_tests.testTools import Progress

NREA = 12  # number of realizations
MAX_ACTORS = 12  # number of nodes (parallel computing)
N = 3000  # series data points


# model parameters
model_params = {
    "hidden_layers": [150, 150, 150],
    "afn": "relu",
}
# training parameters
training_params = {
    "batch_size": "full",
    "max_epochs": 40000,
    "lr": 1e-3,
    "ref_batch_factor": 4,
    "stop_patience": 500,
    "stop_min_delta": 0.0,
    "val_size": 0.2,
}


@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, progress):
    model = HNee(x, **model_params)
    model.fit(**training_params, verbose=False)
    progress.update.remote()
    return model.get_h()


def testHnee01():
    print("MINE methods comparison")
    # gaussian noise parameters
    sigma = 1
    # Generate data
    x = np.random.randn(N) * sigma
    true_h = 0.5 * np.log(2 * np.pi * np.exp(1) * sigma**2)
    x_id = ray.put(x)

    progress = Progress.remote(NREA)
    res = ray.get([model_training.remote(x_id, progress) for _ in range(NREA)])
    results = [("true h", true_h)]
    results += [("hnee", h) for h in res]
    results += [("error", true_h - h) for h in res]
    results = pd.DataFrame(results, columns=["method", "h"])
    sns.catplot(
        data=results,
        x="method",
        y="h",
        # hue="method",
        kind="bar",
        # hue_order=["true h", "h est"],
    )


def testHnee02():
    # Generate data
    sigma = 0.1
    x = np.random.randn(N) * sigma
    true_h = np.log(np.sqrt(2 * np.pi * np.exp(1)) * sigma)
    # models
    model = HNee(x, **model_params)

    model.fit(**training_params, verbose=True)
    # Get h estimation
    h = model.get_h()

    (
        val_loss_epoch,
        val_ema_loss_epoch,
        val_h_epoch,
        test_h_epoch,
    ) = model.get_curves()

    print(f"True h={true_h}, estimated h={h}")
    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    axs[0].plot(val_h_epoch, "r", label="val h")
    axs[0].plot(test_h_epoch, "b", label="test h")
    axs[0].axhline(true_h, color="k", label="true value")
    axs[0].axhline(h, color="g", label="estimated h")
    axs[0].legend(loc="upper right")

    axs[1].plot(val_loss_epoch, "r", label="val loss")
    axs[1].plot(val_ema_loss_epoch, "b", label="val loss smoothed")
    axs[1].legend(loc="upper right")

    fig.suptitle(
        f"true h={true_h:.2f} and estim. h={h:.2f} ",
        fontsize=13,
    )


if __name__ == "__main__":
    testHnee01()
    testHnee02()
    plt.show()
