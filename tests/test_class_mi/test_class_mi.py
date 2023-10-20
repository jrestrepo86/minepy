#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import version

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_mi import ClassMI

N = 10000
# Net parameters
model_params = {"hidden_layers": [32, 16, 8], "afn": "gelu"}
# Training
batch_size = "full"
max_epochs = 2000
train_params = {
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "val_size": 0.2,
    "lr": 1e-4,
    "lr_factor": 0.5,
    "lr_patience": 300,
    "stop_patience": 600,
    "stop_min_delta": 0,
    "weight_decay": 5e-5,
}


def testClassMi01():
    mu = np.array([0, 0])
    Rho = np.linspace(-0.98, 0.98, 21)
    true_mi = np.zeros(*Rho.shape)
    class_mi = np.zeros(*true_mi.shape)

    for i, rho in enumerate(tqdm(Rho)):
        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(
            mean=mu, cov=cov_matrix, size=(N, 1)
        )
        X = np.squeeze(joint_samples_train[:, :, 0])
        Y = np.squeeze(joint_samples_train[:, :, 1])

        # Teoric value
        true_mi[i] = -0.5 * np.log(1 - rho**2)
        # models
        class_mi_model = ClassMI(X, Y, **model_params)
        # Train models
        class_mi_model.fit(**train_params, verbose=False)
        # Get mi estimation
        class_mi[i] = class_mi_model.get_mi()

    # Plot
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.plot(Rho, true_mi, ".k", label="True mi")
    ax.plot(Rho, class_mi, "b", label="Class mi")
    ax.legend(loc="upper center")
    ax.set_xlabel("rho")
    ax.set_ylabel("mi")
    ax.set_title("Classification based mutual information")


def testClassMi02():
    mu = np.array([0, 0])
    rho = 0.95
    true_mi = -0.5 * np.log(1 - rho**2)
    # Generate data
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples_train = np.random.multivariate_normal(
        mean=mu, cov=cov_matrix, size=(N, 1)
    )
    X = np.squeeze(joint_samples_train[:, :, 0])
    Y = np.squeeze(joint_samples_train[:, :, 1])
    # models
    class_mi_model = ClassMI(X, Y, **model_params)

    class_mi_model.fit(**train_params, verbose=True)
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
        f"Curves for rho={rho}, true mi={true_mi:.2f} and estim. mi={class_mi:.2f} ",
        fontsize=13,
    )


if __name__ == "__main__":
    testClassMi01()
    testClassMi02()
    plt.show()
