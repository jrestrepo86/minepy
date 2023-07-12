#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_cmi import ClassCMI


def coupledHenon(n, c):
    """Coupled Henon map X1->X2"""
    n0 = 1000
    n = n + n0
    x = np.zeros((n, 2))
    x[0:2, :] = np.random.rand(2, 2)
    for i in range(2, n):
        x[i, 0] = 1.4 - x[i - 1, 0] ** 2 + 0.3 * x[i - 2, 0]

        x[i, 1] = (
            1.4
            - c * x[i - 1, 0] * x[i - 1, 1]
            - (1 - c) * x[i - 1, 1] ** 2
            + 0.3 * x[i - 2, 1]
        )

    return x[n0:, :]


def testClassCMi02():
    input_dim = 3
    model_params = {
        "hidden_dim": 50,
        "afn": "relu",
        "num_hidden_layers": 3,
    }

    # Generate data
    n = 5000
    c = 0.0
    henon = coupledHenon(n, c)
    u = 2
    X = np.squeeze(henon[: n - u, 0])
    Y = np.squeeze(henon[: n - u, 1])
    Z = np.squeeze(henon[u:, 0])

    # models
    class_cmi_model = ClassCMI(input_dim, **model_params)
    # Train models
    batch_size = 500
    max_epochs = 5000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "knn": 1,
        "lr": 1e-3,
        "lr_factor": 0.1,
        "lr_patience": 10,
        "stop_patience": 100,
        "stop_min_delta": 0.01,
        "verbose": True,
    }

    class_cmi_model.fit(X, Y, Z, **train_params)
    # Get mi estimation
    class_cmi = class_cmi_model.get_mi()
    (
        Dkl_train,
        Dkl_val,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ) = class_cmi_model.get_curves()

    print(f"C={c}, CMI={class_cmi}")
    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    axs[0].plot(Dkl_train, "b", label="Train")
    axs[0].plot(Dkl_val, "r", label="Val")
    axs[0].set_title("Donsker-Varadhan representation")
    axs[0].legend(loc="lower right")

    axs[1].plot(train_loss, "b", label="Train")
    axs[1].plot(val_loss, "r", label="Val")
    axs[1].set_title("Cross-Entropy loss")

    axs[2].plot(train_acc, "b", label="Train")
    axs[2].plot(val_acc, "r", label="Val")
    axs[2].set_title("Binary classifier accuracy")
    axs[2].set_xlabel("Epochs")
    # fig.suptitle(
    #     f"Curves for rho={rho}, true mi={mi_teo:.2f} and estim. mi={class_mi:.2f} ",
    #     fontsize=13,
    # )


if __name__ == "__main__":
    # testClassMi01()
    testClassCMi02()
    plt.show()
