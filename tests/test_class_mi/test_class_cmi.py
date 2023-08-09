#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_cmi_diff import ClassCMIDiff


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
    # Generate data
    n = 10000
    c = 0.8
    henon = coupledHenon(n, c)
    u = 1
    X = np.squeeze(henon[u:, 1])  # target
    Y = np.squeeze(henon[: n - u, 0])  # source
    Z = np.squeeze(henon[: n - u, 1])  # target history

    # models
    model_params = {
        "hidden_dim": 150,
        "afn": "elu",
        "num_hidden_layers": 3,
    }
    class_cmi_model = ClassCMIDiff(X, Y, Z, **model_params)
    # Train models
    batch_size = 150
    max_epochs = 5000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-3,
        "lr_factor": 0.5,
        "lr_patience": 1000,
        "stop_patience": 100,
        "stop_min_delta": 0.01,
        "verbose": True,
    }

    class_cmi_model.fit(**train_params)
    # Get mi estimation
    class_cmi = class_cmi_model.get_cmi()
    (
        cmi_train,
        cmi_val,
        Dkl_train_xyz,
        Dkl_val_xyz,
        train_loss_xyz,
        val_loss_xyz,
        train_acc_xyz,
        val_acc_xyz,
        Dkl_train_xz,
        Dkl_val_xz,
        train_loss_xz,
        val_loss_xz,
        train_acc_xz,
        val_acc_xz,
    ) = class_cmi_model.get_curves()

    print(f"C={c}, CMI={class_cmi}, CMI2={Dkl_val_xyz[-1] - Dkl_val_xz[-1]}")
    # print(f"C={c}, CMI={class_cmi}")
    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    axs[0].plot(Dkl_val_xyz, "b", label="XYZ")
    axs[0].plot(Dkl_val_xz, "r", label="XZ")
    axs[0].set_title("Donsker-Varadhan representation")
    axs[0].legend(loc="lower right")

    axs[1].plot(val_acc_xyz, "b", label="XYZ")
    axs[1].plot(val_acc_xz, "r", label="XZ")
    axs[1].set_title("Accuracy")
    axs[2].plot(val_loss_xyz, "b", label="XYZ")
    axs[2].plot(val_loss_xz, "r", label="XZ")
    axs[2].set_title("Cross-Entropy loss")
    #
    # axs[2].plot(train_acc, "b", label="Train")
    # axs[2].plot(val_acc, "r", label="Val")
    # axs[2].set_title("Binary classifier accuracy")
    # axs[2].set_xlabel("Epochs")
    # fig.suptitle(
    #     f"Curves for rho={rho}, true mi={mi_teo:.2f} and estim. mi={class_mi:.2f} ",
    #     fontsize=13,
    # )


if __name__ == "__main__":
    # testClassMi01()
    testClassCMi02()
    plt.show()
