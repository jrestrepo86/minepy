#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from minepy.mineLayers import T1 as T
from minepy.minepy import Mine

# Datos
n = 100000
x = np.random.rand(1, n) * 2 * np.pi - np.pi
y = np.sin(x)

# net
loss = 'mine'
input_dim = 2
Net = T(input_dim, dim_feedforward=20)
# mine model
model_ft1 = Mine(Net, loss=loss, alpha=0.01)

# parameters
nEpoch = 1000
bSize = 64

_, mi1 = model_ft1.optimize(x, y, batchSize=bSize, numEpochs=nEpoch)

plt.plot(mi1, 'k')
plt.show()
