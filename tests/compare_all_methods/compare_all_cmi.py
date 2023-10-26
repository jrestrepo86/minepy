#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from typing import Counter

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.class_mi.class_diff_cmi import ClassDiffCMI
from minepy.class_mi.class_gen_cmi import ClassGenCMI
from minepy.gan_mi.gan_cmi import GanCMI

DATA_PATH = "../data_cmi"
FILES = {
    "lf_20kdz10": {
        "data": "Linear_Data/catF/data.20k.dz10.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz10.npy",
    },
    # "lf_5kdz20": {
    #     "data": "Linear_Data/catF/data.5k.dz20.seed0.npy",
    #     "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    # },
    # "lf_10kdz20": {
    #     "data": "Linear_Data/catF/data.10k.dz20.seed0.npy",
    #     "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    # },
    # "lf_20kdz100": {
    #     "data": "Linear_Data/catF/data.20k.dz100.seed0.npy",
    #     "ksg": "Linear_Data/catF/ksg_gt.dz100.npy",
    # },
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
    "nl_10kdz20": {
        "data": "Non_Linear_Data/catNon-lin-NI_6/data.10k.dz20.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_6/ksg_gt.dz20.npy",
    },
    # "nl_5kdz100": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_13/data.5k.dz100.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_13/ksg_gt.dz100.npy",
    # },
    # "nl_10kdz100": {
    #     "data": "Non_Linear_Data/catNon-lin-NI_14/data.10k.dz100.seed0.npy",
    #     "ksg": "Non_Linear_Data/catNon-lin-NI_14/ksg_gt.dz100.npy",
    # },
}

NREA = 5  # number of realizations
MAX_ACTORS = 5  # number of nodes
METHODS = ["ccmi-gen", "ccmi-diff", "c-mi-gan"]

# model parameters
afn = "gelu"

# training parameters
general_training_params = {
    "batch_size": "full",
    "max_epochs": 20000,
    "lr": 1e-6,
    "weight_decay": 1e-5,
    "stop_patience": 1000,
    "stop_min_delta": 0.0,
    "verbose": False,
}


@ray.remote
class Progress:
    def __init__(self, max_it=1):
        self.max_it = max_it
        self.count = 0

    def update(self):
        self.count += 1
        print(f"progress: {100* self.count/self.max_it:.2f}%")


@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, z, sim, method, model_params, training_params, progress):
    if method == "ccmi-gen":
        model = ClassGenCMI(x, y, z, **model_params)
    elif method == "ccmi-diff":
        model = ClassDiffCMI(x, y, z, **model_params)
    else:
        model = GanCMI(x, y, z, **model_params)
    model.fit(**training_params)
    progress.update.remote()
    return (sim, method, model.get_cmi())


def read_data(key):
    data = np.load(f"{DATA_PATH}/{FILES[key]['data']}")
    true_cmi = np.load(f"{DATA_PATH}/{FILES[key]['ksg']}")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    return x, y, z, true_cmi[0]


def set_model_parameters(method, dimz):
    if method == "ccmi-gen":
        gdim = 2 + dimz
        model_params = {
            "hidden_layers": [gdim, gdim / 2, gdim / 4],
            "afn": afn,
        }
        training_params = {
            **general_training_params,
            "val_size": 0.2,
        }
    elif method == "ccmi-diff":
        gdim = 2 + dimz
        model_params = {
            "hidden_layers_xyz": [gdim, gdim / 2, gdim / 4],
            "hidden_layers_xz": [gdim, gdim / 2, gdim / 4],
            "afn": afn,
        }
        training_params = {
            **general_training_params,
            "val_size": 0.2,
        }
    else:
        noise_dim = 80
        gdim = noise_dim + dimz
        model_params = {
            "noise_dim": noise_dim,
            "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
            "g_afn": afn,
            "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
            "r_afn": afn,
        }
        training_params = {
            **general_training_params,
            "r_training_steps": 10,
            "g_training_steps": 1,
        }
    return model_params, training_params


def cmiTest():
    print("All methods comparison")
    sims = FILES.keys()
    results = []
    sims_params = []
    # Simulations
    for sim in sims:
        # data
        x, y, z, true_cmi = read_data(sim)
        x_id = ray.put(x)
        y_id = ray.put(y)
        z_id = ray.put(z)
        results += [(sim, "true value", true_cmi)]
        for method in METHODS:
            model_params, training_params = set_model_parameters(method, z.shape[1])
            for _ in range(NREA):
                sim_params_ = {
                    "x": x_id,
                    "y": y_id,
                    "z": z_id,
                    "sim": sim,
                    "method": method,
                    "model_params": model_params,
                    "training_params": training_params,
                }
                sims_params.append(sim_params_)

    progress = Progress.remote(len(sims_params))
    random.shuffle(sims_params)
    res = ray.get(
        [
            model_training.remote(
                s["x"],
                s["y"],
                s["z"],
                s["sim"],
                s["method"],
                s["model_params"],
                s["training_params"],
                progress,
            )
            for s in sims_params
        ]
    )
    # parse results
    results += [(sim_key, method, cmi) for sim_key, method, cmi in res]
    results = pd.DataFrame(results, columns=["sim", "method", "cmi"])
    # save
    results.to_csv("./contrast_all_cmi.csv", sep=",")
    # plot
    sns.catplot(
        data=results,
        x="sim",
        y="cmi",
        hue="method",
        kind="bar",
        hue_order=["true value", "c-mi-gan", "ccmi-gen", "ccmi-diff"],
    )


if __name__ == "__main__":
    cmiTest()
    plt.show()


# def eval_class_gen_model(x, y, z, true_cmi, sim):
#     print("class-gen")
#     gdim = 2 + z.shape[1]
#     class_model_params = {
#         "hidden_layers": [gdim, gdim / 2, gdim / 4],
#         "afn": afn,
#     }
#     class_training_params = {
#         **general_training_params,
#         "val_size": 0.2,
#     }
#     class_cmi_gen_model = ClassGenCMI(x, y, z, **class_model_params)
#     cmi = calc_cmi(class_cmi_gen_model, class_training_params)
#     temp = [(sim, true_cmi, "true value")]
#     temp += [(sim, x, "ccmi-gen") for x in cmi]
#     temp = pd.DataFrame(temp, columns=["sim", "cmi", "method"])
#     return temp
#
#
# def eval_class_diff_model(x, y, z, true_cmi, sim):
#     print("class-diff")
#     gdim = 2 + z.shape[1]
#     class_model_params = {
#         "hidden_layers_xyz": [gdim, gdim / 2, gdim / 4],
#         "hidden_layers_xz": [gdim, gdim / 2, gdim / 4],
#         "afn": afn,
#     }
#     class_training_params = {
#         **general_training_params,
#         "val_size": 0.2,
#     }
#     class_cmi_diff_model = ClassDiffCMI(x, y, z, **class_model_params)
#     cmi = calc_cmi(class_cmi_diff_model, class_training_params)
#     temp = [(sim, true_cmi, "true value")]
#     temp += [(sim, x, "ccmi-diff") for x in cmi]
#     temp = pd.DataFrame(temp, columns=["sim", "cmi", "method"])
#     return temp
#
#
# def eval_gan_model(x, y, z, true_cmi, sim):
#     print("c-mi-gan")
#     noise_dim = 80
#     gdim = noise_dim + z.shape[1]
#     gan_model_params = {
#         "noise_dim": noise_dim,
#         "g_hidden_layers": [gdim, gdim / 2, gdim / 4],
#         "g_afn": afn,
#         "r_hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
#         "r_afn": afn,
#     }
#     gan_training_params = {
#         **general_training_params,
#         "r_training_steps": 10,
#         "g_training_steps": 1,
#     }
#     cmi_gan_model = GanCMI(x, y, z, **gan_model_params)
#     cmi = calc_cmi(cmi_gan_model, gan_training_params)
#     temp = [(sim, true_cmi, "true value")]
#     temp += [(sim, x, "c-mi-gan") for x in cmi]
#     temp = pd.DataFrame(temp, columns=["sim", "cmi", "method"])
#     return temp
