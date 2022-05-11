#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import minepy.mineLayers as Layers
from minepy.minepy import Mine

matplotlib.use('Qt5Agg')


def plotSim01(ax, Rho, TrueMi, MI, label):
    ax.plot(Rho, TrueMi, '.k', label='True')
    ax.plot(Rho, MI, 'b', label=label)
    ax.legend(loc='upper center')
    ax.set_title(label)


def plotSim02(ax, miTeo, miMine, EpochMI, label):
    ax.plot(EpochMI, 'b', label=label)
    ax.axhline(y=miTeo, color='r', linestyle='-')
    ax.axhline(y=miMine, color='k', linestyle='-')
    # ax.legend(loc='upper center')
    ax.set_title(label)


def simMutualInfo():

    # Net
    loss1 = 'mine_biased'
    loss2 = 'mine'
    loss3 = 'remine'
    loss4 = 'clip'
    model_ft1 = Mine(input_dim=2, loss=loss1)
    model_ft2 = Mine(input_dim=2, loss=loss2, alpha=0.01)
    model_ft3 = Mine(input_dim=2, loss=loss3, regWeight=1, targetVal=0)
    model_ft4 = Mine(input_dim=2, loss=loss4, clip=15)

    mu = np.array([0, 0])
    Rho = np.linspace(-0.99, 0.99, 21)
    mi = np.zeros(*Rho.shape)
    mi1 = np.zeros(*mi.shape)
    mi2 = np.zeros(*mi.shape)
    mi3 = np.zeros(*mi.shape)
    mi4 = np.zeros(*mi.shape)

    # Training
    bSize = 300
    nEpoch = 300
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
        mi1[i], _ = model_ft1.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        mi2[i], _ = model_ft2.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        mi3[i], _ = model_ft3.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        mi4[i], _ = model_ft4.optimize(X, Z, batchSize=bSize, numEpochs=nEpoch)
        model_ft1.netReset()
        model_ft2.netReset()
        model_ft3.netReset()
        model_ft4.netReset()

    # Plot
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plotSim01(axs[0, 0], Rho, mi, mi1, label=loss1)
    plotSim01(axs[0, 1], Rho, mi, mi2, label=loss2)
    plotSim01(axs[1, 0], Rho, mi, mi3, label=loss3)
    plotSim01(axs[1, 1], Rho, mi, mi4, label=loss4)
    plt.show()


def simVariance():
    # Net
    afn = 'relu'
    loss1 = 'mine_biased'
    loss2 = 'mine'
    loss3 = 'remine'
    loss4 = 'clip'
    model_ft1 = Mine(input_dim=2, loss=loss1, afn=afn)
    model_ft2 = Mine(input_dim=2, loss=loss2, afn=afn, alpha=0.01)
    model_ft3 = Mine(input_dim=2,
                     loss=loss3,
                     afn=afn,
                     regWeight=4,
                     targetVal=2)
    model_ft4 = Mine(input_dim=2, loss=loss4, afn=afn, clip=10)

    # Generate data
    mu = np.array([0, 0])
    rho = 0.8
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples_train = np.random.multivariate_normal(mean=mu,
                                                        cov=cov_matrix,
                                                        size=(10000, 1))
    X = np.squeeze(joint_samples_train[:, :, 0])
    Z = np.squeeze(joint_samples_train[:, :, 1])
    # Teoric value
    mi = -0.5 * np.log(1 - rho**2)
    # Training
    bSize = 500
    nEpoch = 500

    # Train model
    print(f'Modelo {loss1}')
    m1, epochMI1 = model_ft1.optimize(X,
                                      Z,
                                      batchSize=bSize,
                                      numEpochs=nEpoch,
                                      disableTqdm=False)
    print(f'Modelo {loss2}')
    m2, epochMI2 = model_ft2.optimize(X,
                                      Z,
                                      batchSize=bSize,
                                      numEpochs=nEpoch,
                                      disableTqdm=False)
    print(f'Modelo {loss3}')
    m3, epochMI3 = model_ft3.optimize(X,
                                      Z,
                                      batchSize=bSize,
                                      numEpochs=nEpoch,
                                      disableTqdm=False)
    print(f'Modelo {loss4}')
    m4, epochMI4 = model_ft4.optimize(X,
                                      Z,
                                      batchSize=bSize,
                                      numEpochs=nEpoch,
                                      disableTqdm=False)

    # Plot
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    plotSim02(axs[0, 0], mi, m1, epochMI1, label=loss1)
    plotSim02(axs[0, 1], mi, m2, epochMI2, label=loss2)
    plotSim02(axs[1, 0], mi, m3, epochMI3, label=loss3)
    plotSim02(axs[1, 1], mi, m4, epochMI4, label=loss4)
    print(f'm1={mi}, mi1={m1}, mi2={m2}, mi3={m3}, mi4={m4}')
    plt.show()


if __name__ == "__main__":
    simMutualInfo()
    # simVariance()
