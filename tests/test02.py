#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

import minepy.mineLayers as Layers
from minepy.minepy import Mine

# parameters
bSize = 64
nEpoch = 500

# net
input_dim = 2
Net1 = Layers.T1(input_dim, dim_feedforward=20)
Net2 = Layers.T1(input_dim, dim_feedforward=20)
Net3 = Layers.T1(input_dim, dim_feedforward=20)
# mine model
model_ft1 = Mine(Net1, loss='mine', alpha=0.01)
model_ft2 = Mine(Net2, loss='mine', alpha=1)
model_ft3 = Mine(Net3, loss='mine', alpha=0.0001)
# optimizer

# Generate data
mu = np.array([0, 0])
rho = 0.5
mi = np.ones((nEpoch, 1)) * (-0.5 * np.log(1 - rho**2))
mi1 = np.zeros((nEpoch, 1))
mi2 = np.zeros((nEpoch, 1))
mi3 = np.zeros((nEpoch, 1))
mi4 = np.zeros((nEpoch, 1))

cov_matrix = np.array([[1, rho], [rho, 1]])
joint_samples_train = np.random.multivariate_normal(mean=mu,
                                                    cov=cov_matrix,
                                                    size=(1000, 1))
X_train = np.squeeze(joint_samples_train[:, :, 0])
Z_train = np.squeeze(joint_samples_train[:, :, 1])

# train model
_, mi1 = model_ft1.optimize(X_train,
                            Z_train,
                            batchSize=bSize,
                            numEpochs=nEpoch)
_, mi2 = model_ft2.optimize(X_train,
                            Z_train,
                            batchSize=bSize,
                            numEpochs=nEpoch)
_, mi3 = model_ft3.optimize(X_train,
                            Z_train,
                            batchSize=bSize,
                            numEpochs=nEpoch)

print('ture MI {0}'.format(mi[0]))
plt.plot(mi1, 'b', label='alfa 0.01')
plt.plot(mi2, 'r', label='0.5')
plt.plot(mi3, 'g', label='0.0001')
plt.plot(mi, 'k', label='true MI')
plt.legend()
plt.show()
