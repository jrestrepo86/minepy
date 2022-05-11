#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

from minepy.mineTools import henon
from minepy.tenee import Tenee

# net
# mine model
m = 3
tau = 1
model_ft = Tenee(m=m, tau=tau, u=tau, dim_feedforward=100, device=None)

# Generate data
c = 0.5
n = 10000
data = henon(n, c)

X = np.squeeze(data[:, 0])
Z = np.squeeze(data[:, 1])

# train model
bSize = 1024
nEpoch = 2024

TE, Epoch_train = model_ft.optimize(X,
                                    Z,
                                    batchSize=bSize,
                                    batchScale=50,
                                    numEpochs=nEpoch,
                                    disableTqdm=False)

(TE_train, N1_train, N2_train, N3_train, N4_train) = Epoch_train
print(f'TE = {TE}')
plt.plot(TE_train, 'b', label='TE')
plt.plot(N1_train, 'g', label='N1')
plt.plot(N2_train, 'r', label='N2')
plt.plot(N3_train, 'k', label='N3')
plt.plot(N4_train, 'c', label='N4')
plt.legend()
plt.show()
