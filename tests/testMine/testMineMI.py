#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import minepy.mineLayers as Layers
from minepy.minepy import Mine

matplotlib.use("Qt5Agg")


def plotSim01(ax, Rho, TrueMi, MI, label):
    ax.plot(Rho, TrueMi, ".k", label="True")
    ax.plot(Rho, MI, "b", label=label)
    ax.legend(loc="upper center")
    ax.set_title(label)


def simMutualInfo():

    # Net
    loss1 = "mine_biased"
    loss2 = "mine"
    loss3 = "remine"
    loss4 = 'remine'
    model_ft1 = Mine(input_dim=2, loss=loss1)
    model_ft2 = Mine(input_dim=2, loss=loss2, alpha=0.1)
    model_ft3 = Mine(input_dim=2, loss=loss3, regWeight=1, targetVal=0.0)
    model_ft4 = Mine(input_dim=2, loss=loss3, regWeight=0.1, targetVal=0.0)
    # model_ft4 = Mine(input_dim=2, loss=loss4, clip=15)

    mu = np.array([0, 0])
    Rho = np.linspace(-0.99, 0.99, 21)
    mi = np.zeros(*Rho.shape)
    mi1 = np.zeros(*mi.shape)
    mi2 = np.zeros(*mi.shape)
    mi3 = np.zeros(*mi.shape)
    mi4 = np.zeros(*mi.shape)

    # Training
    bSize = 1000
    nEpoch = 1000
    for i, rho in enumerate(tqdm(Rho)):

        model_ft1 = Mine(input_dim=2, loss=loss1)
        model_ft2 = Mine(input_dim=2, loss=loss2, alpha=0.1)
        model_ft3 = Mine(input_dim=2, loss=loss3, regWeight=1, targetVal=0.0)
        model_ft4 = Mine(input_dim=2, loss=loss3, regWeight=0.1, targetVal=0.0)
        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(mean=mu,
                                                            cov=cov_matrix,
                                                            size=(10000, 1))
        X = np.squeeze(joint_samples_train[:, :, 0])
        Z = np.squeeze(joint_samples_train[:, :, 1])

        # Teoric value
        mi[i] = -0.5 * np.log(1 - rho**2)
        # Train model
        mi1[i], _ = model_ft1.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        mi2[i], _ = model_ft2.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        mi3[i], _ = model_ft3.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        mi4[i], _ = model_ft4.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        # model_ft1.netReset()
        # model_ft2.netReset()
        # model_ft3.netReset()
        # model_ft4.netReset()

    # Plot
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plotSim01(axs[0, 0], Rho, mi, mi1, label=loss1)
    plotSim01(axs[0, 1], Rho, mi, mi2, label=loss2)
    plotSim01(axs[1, 0], Rho, mi, mi3, label=loss3)
    plotSim01(axs[1, 1], Rho, mi, mi4, label=loss4 + '_0.1')
    plt.show()


if __name__ == "__main__":
    simMutualInfo()