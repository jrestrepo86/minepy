#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_mi import ClassMI


def plot(ax, Rho, teo_mi, mi, label):
    ax.plot(Rho, teo_mi, '.k', label='True')
    ax.plot(Rho, mi, 'b', label=label)
    ax.legend(loc='upper center')
    ax.set_title(label)


def testMine():

    # Net
    input_dim = 2
    model_params = {'hidden_dim': 50, 'afn': 'relu', 'num_hidden_layers': 3}

    mu = np.array([0, 0])
    Rho = np.linspace(-0.99, 0.99, 21)
    mi_teo = np.zeros(*Rho.shape)
    class_mi = np.zeros(*mi_teo.shape)

    # Training
    batch_size = 300
    max_epochs = 3000
    for i, rho in enumerate(tqdm(Rho)):

        # Generate data
        cov_matrix = np.array([[1, rho], [rho, 1]])
        joint_samples_train = np.random.multivariate_normal(mean=mu,
                                                            cov=cov_matrix,
                                                            size=(10000, 1))
        X = np.squeeze(joint_samples_train[:, :, 0])
        Z = np.squeeze(joint_samples_train[:, :, 1])

        # Teoric value
        mi_teo[i] = -0.5 * np.log(1 - rho**2)
        # models
        class_mi_model = ClassMI(input_dim, **model_params)
        # Train models
        class_mi_model.fit(X, Z, batch_size=batch_size, max_epochs=max_epochs)
        # Get mi estimation
        class_mi[i] = class_mi_model.calc_mi_fn(X, Z)

    # # Plot
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    plot(axs, Rho, mi_teo, class_mi, label='')
    plt.show()


if __name__ == "__main__":
    testMine()
