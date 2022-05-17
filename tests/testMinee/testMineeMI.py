#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.minee import Minee

matplotlib.use('Qt5Agg')


def plotSim01(ax, Rho, TrueMi, MI, label):
    ax.plot(Rho, TrueMi, '.k', label='True')
    ax.plot(Rho, MI, 'b', label=label)
    ax.legend(loc='upper center')
    ax.set_title(label)


def simMutualInfo():

    # Net
    model_ft1 = Minee(
        input_dim=1,
        afn='relu',
        hidden_dim=50,
        nLayers=2,
    )
    model_ft2 = Minee(
        input_dim=1,
        afn='relu',
        hidden_dim=50,
        nLayers=2,
    )

    mu = np.array([0, 0])
    Rho = np.linspace(-0.99, 0.99, 21)
    mi = np.zeros(*Rho.shape)
    mi1 = np.zeros(*mi.shape)
    mi2 = np.zeros(*mi.shape)

    # Training
    bSize = 500
    nEpoch = 500
    bScale = 1
    for i, rho in enumerate(tqdm(Rho)):

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
        vals1, _ = model_ft1.optimize(X,
                                      Z,
                                      batchSize=bSize,
                                      numEpochs=nEpoch,
                                      batchScale=bScale)
        vals2, _ = model_ft2.optimize(X,
                                      Z,
                                      batchSize=bSize,
                                      numEpochs=nEpoch,
                                      batchScale=4 * bScale)
        mi1[i] = vals1[-1]
        mi2[i] = vals2[-1]
        model_ft1.netReset()
        model_ft2.netReset()

    # Plot
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    plotSim01(axs[0], Rho, mi, mi1, label='Bscale')
    plotSim01(axs[1], Rho, mi, mi2, label='4*Bscale')
    # plotSim01(axs[1, 1], Rho, mi, mi4, label=loss4)
    plt.show()


if __name__ == "__main__":
    simMutualInfo()
