#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_mi import ClassMI


def testClassMi01():
    # Net parameters
    model_params = {"hidden_dim": 50, "afn": "relu", "num_hidden_layers": 3}

    mu = np.array([0, 0])
    Rho = np.linspace(-0.98, 0.98, 21)
    mi_teo = np.zeros(*Rho.shape)
    class_mi = np.zeros(*mi_teo.shape)

    # Training
    batch_size = 300
    max_epochs = 3000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-3,
        "lr_factor": 0.5,
        "lr_patience": 30,
        "stop_patience": 100,
        "stop_min_delta": 0.01,
        "verbose": False,
    }
    for i, rho in enumerate(tqdm(Rho)):
        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(
            mean=mu, cov=cov_matrix, size=(10000, 1)
        )
        X = np.squeeze(joint_samples_train[:, :, 0])
        Y = np.squeeze(joint_samples_train[:, :, 1])

        # Teoric value
        mi_teo[i] = -0.5 * np.log(1 - rho**2)
        # models
        class_mi_model = ClassMI(X, Y, **model_params)
        # Train models
        class_mi_model.fit(**train_params)
        # Get mi estimation
        class_mi[i] = class_mi_model.get_mi()

    # Plot
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.plot(Rho, mi_teo, ".k", label="True mi")
    ax.plot(Rho, class_mi, "b", label="Class mi")
    ax.legend(loc="upper center")
    ax.set_xlabel("rho")
    ax.set_ylabel("mi")
    ax.set_title("Classification based mutual information")


def testClassMi02():
    model_params = {"hidden_dim": 50, "afn": "relu", "num_hidden_layers": 3}
    mu = np.array([0, 0])
    rho = 0.95
    mi_teo = -0.5 * np.log(1 - rho**2)
    # Generate data
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples_train = np.random.multivariate_normal(
        mean=mu, cov=cov_matrix, size=(10000, 1)
    )
    X = np.squeeze(joint_samples_train[:, :, 0])
    Y = np.squeeze(joint_samples_train[:, :, 1])
    # models
    class_mi_model = ClassMI(X, Y, **model_params)
    # Train models
    batch_size = 300
    max_epochs = 3000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-3,
        "lr_factor": 0.5,
        "lr_patience": 30,
        "stop_patience": 100,
        "stop_min_delta": 0.01,
        "verbose": True,
    }

    class_mi_model.fit(**train_params)
    # Get mi estimation
    class_mi = class_mi_model.get_mi()
    Dkl_val, val_loss, val_acc = class_mi_model.get_curves()

    print(f"MI={mi_teo}, MI_class={class_mi}")
    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    axs[0].plot(Dkl_val, "r", label="Val")
    axs[0].set_title("Donsker-Varadhan representation")
    axs[0].legend(loc="lower right")

    axs[1].plot(val_loss, "r", label="Val")
    axs[1].set_title("Cross-Entropy loss")

    axs[2].plot(val_acc, "r", label="Val")
    axs[2].set_title("Binary classifier accuracy")
    axs[2].set_xlabel("Epochs")
    fig.suptitle(
        f"Curves for rho={rho}, true mi={mi_teo:.2f} and estim. mi={class_mi:.2f} ",
        fontsize=13,
    )


if __name__ == "__main__":
    testClassMi01()
    testClassMi02()
    plt.show()
