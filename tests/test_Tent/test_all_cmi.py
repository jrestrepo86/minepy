#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_diff_cmi import ClassDiffCMI
from minepy.class_mi.class_gen_cmi import ClassGenCMI
from minepy.gan_mi.gan_cmi import GanCMI

#!/usr/bin/env python
# -*- coding: utf-8 -*-



DATA_PATH = "../data_cmi"
FILES = {
    "lf_20kdz10": {
        "data": "Linear_Data/catF/data.20k.dz10.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz10.npy",
    },
    "lf_5kdz20": {
        "data": "Linear_Data/catF/data.5k.dz20.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    },
    "lf_10kdz20": {
        "data": "Linear_Data/catF/data.10k.dz20.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    },
    "lf_20kdz100": {
        "data": "Linear_Data/catF/data.20k.dz100.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz100.npy",
    },
    "nl_5kdz10": {
        "data": "Non_Linear_Data/catNon-lin-NI_1/data.5k.dz10.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_1/ksg_gt.dz10.npy",
    },
    "nl_10kdz10": {
        "data": "Non_Linear_Data/catNon-lin-NI_2/data.10k.dz10.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_2/ksg_gt.dz10.npy",
    },
    "nl_5kdz20": {
        "data": "Non_Linear_Data/catNon-lin-NI_5/data.5k.dz20.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_5/ksg_gt.dz20.npy",
    },
    "nl_10kdz20": {
        "data": "Non_Linear_Data/catNon-lin-NI_6/data.10k.dz20.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_6/ksg_gt.dz20.npy",
    },
    "nl_5kdz100": {
        "data": "Non_Linear_Data/catNon-lin-NI_13/data.5k.dz100.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_13/ksg_gt.dz100.npy",
    },
    "nl_10kdz100": {
        "data": "Non_Linear_Data/catNon-lin-NI_14/data.10k.dz100.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_14/ksg_gt.dz100.npy",
    },
}


def read_data(key):
    data = np.load(f"{DATA_PATH}/{FILES[key]['data']}")
    true_cmi = np.load(f"{DATA_PATH}/{FILES[key]['ksg']}")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    return x, y, z, true_cmi[0]


def get_cmi(x, y, z, model_params, training_params, key):
    cmi = []
    for _ in range(5):
        class_cmigan_model = GanCMI(x, y, z, **model_params)
        class_cmigan_model.fit(**training_params, verbose=False)
        cmi.append(class_cmigan_model.get_cmi())
    return cmi


def cmiLinear():
    # data
    data_keys = FILES.keys()

    # model parameters
    max_epochs = 5000
    batch_size = "full"
    noise_dim = 80

    # Training
    training_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "lr": 1e-4,
        "lr_factor": 0.1,
        "lr_patience": 1000,
        "r_training_steps": 5,
        "g_training_steps": 1,
        "weight_decay": 1e-5,
    }
    results = []
    for key in data_keys:
        print(key)
        x, y, z, true_cmi = read_data(key)
        # Net parameters
        gdim = noise_dim + z.shape[1]
        model_params = {
            "noise_dim": noise_dim,
            "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
            "g_afn": "gelu",
            "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
            "r_afn": "gelu",
        }
        cmi = get_cmi(x, y, z, model_params, training_params, key)
        temp = [(key, true_cmi, "true val")]
        temp += [(key, x, "estimated") for x in cmi]
        temp = pd.DataFrame(temp, columns=["sim", "cmi", "v"])
        results.append(temp)
    results = pd.concat(results, ignore_index=True)
    sns.catplot(data=results, x="sim", y="cmi", hue="v", kind="bar")
    # fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
    # epoch = np.arange(cmi_epoch.size)
    # axs[0].set_title("")
    # axs[0].plot(epoch, cmi_epoch, "r", label="")
    # axs[0].axhline(cmi_teo, color="b", label="")
    # axs[0].axhline(cmi, color="k", label="")
    # axs[1].plot(epoch, gen_loss_epoch, "b", label="gen")
    # axs[1].plot(epoch, reg_marg_loss, "r", label="gen")
    # axs[2].plot(epoch, reg_loss_epoch, "r", label="reg")
    plt.show()


if __name__ == "__main__":
    cmiLinear()
