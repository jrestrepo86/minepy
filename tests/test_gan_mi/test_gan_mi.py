#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import version

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.gan_mi.gan_mi import GanMI

# sample points
N = 3000
# Net parameters
noise_dim = 80
gdim = noise_dim
model_params = {
    "noise_dim": noise_dim,
    "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
    "g_afn": "gelu",
    "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
    "r_afn": "gelu",
}
# Training
batch_size = "full"
max_epochs = 2000
training_params = {
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "lr": 1e-4,
    "lr_factor": 0.1,
    "lr_patience": 1000,
    "r_training_steps": 5,
    "g_training_steps": 1,
    "weight_decay": 1e-5,
}


def testGanMi01():
    mu = np.array([0, 0])
    Rho = np.linspace(-0.98, 0.98, 21)
    mi_teo = np.zeros(*Rho.shape)
    gan_mi = np.zeros(*mi_teo.shape)

    for i, rho in enumerate(tqdm(Rho)):
        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(
            mean=mu, cov=cov_matrix, size=(N, 1)
        )
        X = np.squeeze(joint_samples_train[:, :, 0])
        Y = np.squeeze(joint_samples_train[:, :, 1])

        # Teoric value
        mi_teo[i] = -0.5 * np.log(1 - rho**2)
        # models
        gan_mi_model = GanMI(X, Y, **model_params)
        # Train models
        gan_mi_model.fit(**training_params, verbose=False)
        # Get mi estimation
        gan_mi[i] = gan_mi_model.get_mi()

    # Plot
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.plot(Rho, mi_teo, ".k", label="True mi")
    ax.plot(Rho, gan_mi, "b", label="Gan mi")
    ax.legend(loc="upper center")
    ax.set_xlabel("rho")
    ax.set_ylabel("mi")
    ax.set_title("Gan based mutual information")


def testGanMi02():
    mu = np.array([0, 0])
    rho = 0.95
    mi_teo = -0.5 * np.log(1 - rho**2)
    # Generate data
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples_train = np.random.multivariate_normal(
        mean=mu, cov=cov_matrix, size=(N, 1)
    )
    X = np.squeeze(joint_samples_train[:, :, 0])
    Y = np.squeeze(joint_samples_train[:, :, 1])
    # models
    gan_mi_model = GanMI(X, Y, **model_params)

    gan_mi_model.fit(**training_params, verbose=True)
    # Get mi estimation
    gan_mi = gan_mi_model.get_mi()
    Dkl_val, gen_loss, reg_loss = gan_mi_model.get_curves()

    print(f"MI={mi_teo}, MI_class={gan_mi}")
    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    axs[0].plot(Dkl_val, "r", label="Val")
    axs[0].set_title("Donsker-Varadhan representation")
    axs[0].legend(loc="lower right")
    axs[1].plot(gen_loss, "r")
    axs[1].set_title("Generator loss")
    axs[2].plot(reg_loss, "r")
    axs[2].set_title("Regresor loss")

    fig.suptitle(
        f"Curves for rho={rho}, true mi={mi_teo:.2f} and estim. mi={gan_mi:.2f} ",
        fontsize=13,
    )


if __name__ == "__main__":
    testGanMi01()
    # testGanMi02()
    plt.show()
