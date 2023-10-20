#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

# from minepy.class_mi.class_diff_cmi import ClassDiffCMI
from minepy.class_mi.class_gen_cmi import ClassGenCMI
from minepy.gan_mi.gan_cmi import GanCMI

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
    # "lf_10kdz20": {
    #     "data": "Linear_Data/catF/data.10k.dz20.seed0.npy",
    #     "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    # },
    "lf_20kdz100": {
        "data": "Linear_Data/catF/data.20k.dz100.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz100.npy",
    },
    # "nl_5kdz10": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_1/data.5k.dz10.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_1/ksg_gt.dz10.npy",
    # },
    # "nl_10kdz10": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_2/data.10k.dz10.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_2/ksg_gt.dz10.npy",
    # },
    # "nl_5kdz20": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_5/data.5k.dz20.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_5/ksg_gt.dz20.npy",
    # },
    # "nl_10kdz20": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_6/data.10k.dz20.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_6/ksg_gt.dz20.npy",
    # },
    # "nl_5kdz100": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_13/data.5k.dz100.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_13/ksg_gt.dz100.npy",
    # },
    # "nl_10kdz100": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_14/data.10k.dz100.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_14/ksg_gt.dz100.npy",
    # },
}

# training parameters
general_training_params = {
    "batch_size": "full",
    "max_epochs": 5000,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "stop_patience": 2000,
    "stop_min_delta": 0.0,
    "verbose": False,
}


def read_data(key):
    data = np.load(f"{DATA_PATH}/{FILES[key]['data']}")
    true_cmi = np.load(f"{DATA_PATH}/{FILES[key]['ksg']}")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    return x, y, z, true_cmi[0]


def get_cmi(x, y, z, model, training_params):
    cmi = []
    for i in range(3):
        print(i + 1)
        model.fit(**training_params)
        cmi.append(model.get_cmi())
    return cmi


def eval_class_gen_model(x, y, z, true_cmi, sim):
    gdim = 2 + z.shape[1]
    class_model_params = {
        "hidden_layers": [gdim, gdim / 2, gdim / 4],
        "afn": "gelu",
    }
    class_training_params = {
        **general_training_params,
        "val_size": 0.2,
    }
    class_cmi_gen_model = ClassGenCMI(x, y, z, **class_model_params)
    cmi = get_cmi(x, y, z, class_cmi_gen_model, class_training_params)
    temp = [(sim, true_cmi, "true value")]
    temp += [(sim, x, "ccmi-gen") for x in cmi]
    temp = pd.DataFrame(temp, columns=["sim", "cmi", "method"])
    return temp


def eval_gan_model(x, y, z, true_cmi, sim):
    noise_dim = 80
    gdim = noise_dim + z.shape[1]
    gan_model_params = {
        "noise_dim": noise_dim,
        "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
        "g_afn": "gelu",
        "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
        "r_afn": "gelu",
    }
    gan_training_params = {
        **general_training_params,
        "r_training_steps": 5,
        "g_training_steps": 1,
    }
    cmi_gan_model = GanCMI(x, y, z, **gan_model_params)
    cmi = get_cmi(x, y, z, cmi_gan_model, gan_training_params)
    temp = [(sim, true_cmi, "true value")]
    temp += [(sim, x, "c-mi-gan") for x in cmi]
    temp = pd.DataFrame(temp, columns=["sim", "cmi", "method"])
    return temp


def cmiTest():
    # data
    data_keys = FILES.keys()
    results = []
    for key in data_keys:
        print(key)
        x, y, z, true_cmi = read_data(key)
        results.append(eval_class_gen_model(x, y, z, true_cmi, key))
        results.append(eval_gan_model(x, y, z, true_cmi, key))
    results = pd.concat(results, ignore_index=True)
    results.to_csv("./contrast_all_cmi.csv", sep=",")
    sns.catplot(data=results, x="sim", y="cmi", hue="method", kind="bar")


if __name__ == "__main__":
    cmiTest()
    plt.show()
