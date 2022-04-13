import numpy as np
from matplotlib import pyplot as plt

import minepy.mineLayers as Layers
from minepy.minee import Minee
from minepy.minepy import Mine
from minepy.mineTools import Interp

# net
input_dim_x = 1
input_dim_z = 1
Netx = Layers.T11(input_dim_x, dim_feedforward=50)
Netz = Layers.T11(input_dim_z, dim_feedforward=50)
Netxz = Layers.T11(input_dim_x + input_dim_z, dim_feedforward=50)
NetMine = Layers.T1(input_dim_x + input_dim_z, dim_feedforward=50)
# mine model
model_ft1 = Minee(Netx, Netz, Netxz, loss=None)
model_ft2 = Mine(NetMine, loss='mine', alpha=0.01)
# optimizer

# Generate data
mu = np.array([0, 0])
rho = 0.0
mi = -0.5 * np.log(1 - rho**2)

cov_matrix = np.array([[1, rho], [rho, 1]])
joint_samples_train = np.random.multivariate_normal(mean=mu,
                                                    cov=cov_matrix,
                                                    size=(10000, 1))
X = np.squeeze(joint_samples_train[:, :, 0])
Z = np.squeeze(joint_samples_train[:, :, 1])

# indx = np.arsort(X)
# indz = np.argsort(Z)

# X_inter = Interp(X.copy().sort(), 100)
# Z_inter = Interp(Z.copy().sort(), 100)

# train model
bSize = 512
nEpoch = 500
vals, vals_train = model_ft1.optimize(X,
                                      Z,
                                      batchSize=bSize,
                                      batchScale=100,
                                      numEpochs=nEpoch,
                                      disableTqdm=False)
mineMI, Mine_train = model_ft2.optimize(X,
                                        Z,
                                        batchSize=bSize,
                                        numEpochs=nEpoch,
                                        disableTqdm=False)

hx1, hz1, hxz1, mi1 = vals
hx_train, hz_train, hxz_train, mi_train = vals_train

print(f'true MI = {mi}, calc MI = {mi1}, Mine ={mineMI}')
plt.plot(mi_train, 'b', label='mi')
plt.plot(Mine_train, 'r', label='mi')
# plt.plot(hx_train, 'r', label='hx')
# plt.plot(hz_train, 'g', label='hz')
# plt.plot(hxz_train, 'k', label='hxz')
# plt.plot(mi2, 'r', label='0.5')
# plt.plot(mi3, 'g', label='0.0001')
plt.axhline(y=mi, color='k', linestyle='-')
plt.legend()
plt.show()
