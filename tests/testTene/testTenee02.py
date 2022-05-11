#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

from minepy.mineTools import henon
from minepy.tenee02 import Tenee

# net
# mine model
m = 3
tau = 1
model_ft1 = Tenee(m=m, tau=tau, u=tau, dim_feedforward=25, device=None)
model_ft2 = Tenee(m=m, tau=tau, u=tau, dim_feedforward=25, device=None)

# Generate data
n = 10000

# train model
bSize = 512
nEpoch = 3000
TE1 = 0
TE2 = 0

c = 0.5
data = henon(n, c)
X = np.squeeze(data[:, 0])
Z = np.squeeze(data[:, 1])
TE1, Epoch_train1 = model_ft1.train(Z,
                                    X,
                                    batchSize=bSize,
                                    batchScale=1,
                                    numEpochs=nEpoch,
                                    disableTqdm=False)
# c = 0.0
# data = henon(n, c)
# X = np.squeeze(data[:, 0])
# Z = np.squeeze(data[:, 1])
# TE2, Epoch_train2 = model_ft2.train(Z,
#                                     X,
#                                     batchSize=bSize,
#                                     batchScale=1,
#                                     numEpochs=nEpoch,
#                                     disableTqdm=False)

(TE_train1, N1_train1, N2_train1, N3_train1, N4_train1) = Epoch_train1
# (TE_train2, N1_train2, N2_train2, N3_train2, N4_train2) = Epoch_train2
print(f'TE c0.5 = {TE1}, TE c0 ={TE2}')
plt.plot(TE_train1, 'b', label='TE')
plt.plot(N1_train1, 'g', label='N1')
plt.plot(N2_train1, 'r', label='N2')
plt.plot(N3_train1, 'k', label='N3')
plt.plot(N4_train1, 'c', label='N4')
plt.legend()
plt.show()
