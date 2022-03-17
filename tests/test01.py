#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from minepy.mineLayers import T1 as T
from minepy.minepy import Mine

# parameters
bSize = 1024
nEpoch = 300

# net
input_dim = 2
Net = T(input_dim, dim_feedforward=30)
# mine model
model_ft = Mine(Net, loss='mine_biased', alpha=0.01)
model_ft2 = Mine(Net, loss='mine', alpha=0.01)
# optimizer

# Generate data
mu = np.array([0, 0])
Rho = np.linspace(-0.99, 0.99, 21)
mi = np.zeros(*Rho.shape)
mi1 = np.zeros(*Rho.shape)
mi2 = np.zeros(*Rho.shape)

for i, rho in enumerate(Rho):
    cov_matrix = np.array([[1, rho], [rho, 1]])
    joint_samples_train = np.random.multivariate_normal(mean=mu,
                                                        cov=cov_matrix,
                                                        size=(10000, 1))
    X_train = np.squeeze(joint_samples_train[:, :, 0])
    Z_train = np.squeeze(joint_samples_train[:, :, 1])

    # train model
    mi[i] = -0.5 * np.log(1 - rho**2)
    mi1[i] = model_ft.optimize(X_train,
                               Z_train,
                               batchSize=bSize,
                               numEpochs=nEpoch)
    mi2[i] = model_ft2.optimize(X_train,
                                Z_train,
                                batchSize=bSize,
                                numEpochs=nEpoch)

plt.plot(Rho, mi, 'k')
plt.plot(Rho, mi1, 'b')
plt.plot(Rho, mi2, 'r')
plt.show()
