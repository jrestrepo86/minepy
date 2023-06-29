#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_mi import ClassMI


def plot(ax, Rho, teo_mi, mi, label):
    ax.plot(Rho, teo_mi, ".k", label="True")
    ax.plot(Rho, mi, "b", label=label)
    ax.legend(loc="upper center")
    ax.set_title(label)


def testClassMi01():
    # Net
    input_dim = 2
    model_params = {"hidden_dim": 50, "afn": "relu", "num_hidden_layers": 3}

    mu = np.array([0, 0])
    Rho = np.linspace(-0.98, 0.98, 21)
    mi_teo = np.zeros(*Rho.shape)
    class_mi01 = np.zeros(*mi_teo.shape)
    class_mi02 = np.zeros(*mi_teo.shape)

    # Training
    batch_size = 300
    max_epochs = 3000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-3,
        "lr_factor": 0.1,
        "lr_patience": 10,
        "stop_patience": 100,
        "stop_min_delta": 0.01,
        "verbose": True,
    }
    for i, rho in enumerate(tqdm(Rho)):
        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(
            mean=mu, cov=cov_matrix, size=(10000, 1)
        )
        X = np.squeeze(joint_samples_train[:, :, 0])
        Z = np.squeeze(joint_samples_train[:, :, 1])

        # Teoric value
        mi_teo[i] = -0.5 * np.log(1 - rho**2)
        # models
        class_mi_model = ClassMI(input_dim, **model_params)
        # Train models
        class_mi_model.fit(X, Z, **train_params)
        # Get mi estimation
        class_mi01[i], class_mi02[i] = class_mi_model.get_mi()

    # # Plot
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    plot(axs[0], Rho, mi_teo, class_mi01, label="class01")
    plot(axs[1], Rho, mi_teo, class_mi02, label="class02")
    plt.show()


def testClassMi02():
    input_dim = 2
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
    Z = np.squeeze(joint_samples_train[:, :, 1])
    # models
    class_mi_model = ClassMI(input_dim, **model_params)
    # Train models
    batch_size = 300
    max_epochs = 3000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-4,
        "lr_factor": 0.5,
        "lr_patience": 30,
        "stop_patience": 500,
        "stop_min_delta": 0.1,
        "verbose": True,
    }

    class_mi_model.fit(X, Z, **train_params)
    # Get mi estimation
    class_mi, _ = class_mi_model.get_mi()
    (
        Dkl_train,
        Dkl_val,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ) = class_mi_model.get_curves()

    print(f"MI={mi_teo}, MI_class={class_mi}")
    # # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs[0].plot(Dkl_train, "b", label="Dkl-Train")
    axs[0].plot(Dkl_val, "r", label="Dkl-Val")
    axs[1].plot(train_loss, "b", label="Train-loss")
    axs[1].plot(val_loss, "r", label="Val-loss")
    axs[2].plot(train_acc, "b", label="Train-acc")
    axs[2].plot(val_acc, "r", label="Val-acc")
    plt.show()


if __name__ == "__main__":
    testClassMi01()
    testClassMi02()
