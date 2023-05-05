#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.mineLT import Mine


def plotSim01(ax, Rho, TrueMi, MI, label):
    ax.plot(Rho, TrueMi, ".k", label="True")
    ax.plot(Rho, MI, "b", label=label)
    ax.legend(loc="upper center")
    ax.set_title(label)


def simMutualInfo():

    # Net

    mu = np.array([0, 0])
    Rho = np.linspace(-0.98, 0.5, 1)
    mi = np.zeros(*Rho.shape)
    mi1 = np.zeros(*Rho.shape)

    # Training
    bSize = 200
    nEpoch = 3000
    for i, rho in enumerate(tqdm(Rho)):
        model = Mine(input_dim=2)

        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(mean=mu,
                                                            cov=cov_matrix,
                                                            size=(10000, 1))
        X = np.squeeze(joint_samples_train[:, :, 0])
        Z = np.squeeze(joint_samples_train[:, :, 1])
        # Teoric value
        mi[i] = -0.5 * np.log(1 - rho**2)

        model.fit(X, Z, batch_size=bSize, max_epochs=nEpoch, verbose=True)
        mi1[i], val_loss = model.get_mi(X, Z)

    print(f'Tmi={mi}, mine={mi1}')

    plt.plot(val_loss)
    plt.show()
    # # Plot
    # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    # plotSim01(axs, Rho, mi, mi1, '')
    # plt.show()


if __name__ == "__main__":
    simMutualInfo()
