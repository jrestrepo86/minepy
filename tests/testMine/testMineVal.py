#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import minepy.mineLayers as Layers
import minepy.mineTools as mineTools
from minepy.minepy import Mine

matplotlib.use("Qt5Agg")


def plotSim(miTeo, miMine, epoch_train, epoch_val):
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
    axs[0].plot(epoch_train, 'b', label='epoch_train')
    axs[0].axhline(y=miTeo, color='r', linestyle='-', label='True mi')
    axs[0].axhline(y=miMine, color='k', linestyle='-', label='Mine')

    axs[1].plot(epoch_val, 'b', linestyle='dotted', label='epoch_val')
    axs[1].axhline(y=miTeo, color='r', linestyle='-', label='True mi')
    axs[1].axhline(y=miMine, color='k', linestyle='-', label='Mine')

    axs[2].plot(epoch_val - epoch_train, 'b', label='epoch_train')

    plt.legend()


def simMutualInfo():

    # Net
    loss1 = "mine"
    model_ft1 = Mine(
        input_dim=2,
        loss=loss1,
        afn='elu',
        hidden_dim=150,
        nLayers=3,
        alpha=0.01,
        regWeight=1,
        targetVal=0.0,
    )

    # Training
    bSize = 600
    nEpoch = 60000

    # Generate data
    # mu = np.array([0, 0])
    # rho = 0.98
    # cov_matrix = np.array([[1, rho], [rho, 1]])
    # joint_samples_train = np.random.multivariate_normal(mean=mu,
    #                                                     cov=cov_matrix,
    #                                                     size=(8000, 1))

    # X = np.squeeze(joint_samples_train[:, :, 0])
    # Z = np.squeeze(joint_samples_train[:, :, 1])
    H = mineTools.henon(3000, 0.0)
    X = H[:-1, 0]
    Z = H[1:, 1]

    # Teoric value
    # mi = -0.5 * np.log(1 - rho**2)
    mi = 0.0
    # Train model
    mi1, epoch_val, epoch_train, _ = model_ft1.optimize_validate(
        X,
        Z,
        val_size=0.2,
        batchSize=bSize,
        numEpochs=nEpoch,
        disableTqdm=False)

    # Plot
    print(f'mi1={mi1}, true={mi}')
    plotSim(mi, mi1, epoch_train, epoch_val)
    plt.show()


if __name__ == "__main__":
    simMutualInfo()
