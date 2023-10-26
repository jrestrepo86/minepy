#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import ray
import seaborn as sns
from matplotlib import pyplot as plt

from minepy.class_mi.class_gen_cmi import ClassGenCMI

DATA_PATH = "../data_cmi"
FILES = {
    "lf_10kdz20": {
        "data": "Linear_Data/catF/data.10k.dz20.seed0.npy",
        "ksg": "Linear_Data/catF/ksg_gt.dz20.npy",
    },
    "nl_10kdz20": {
        "data": "Non_Linear_Data/catNon-lin-NI_6/data.10k.dz20.seed0.npy",
        "ksg": "Non_Linear_Data/catNon-lin-NI_6/ksg_gt.dz20.npy",
    },
}

NREA = 5  # number of realizations
MAX_ACTORS = 5  # number of nodes

# model parameters
afn = "gelu"

# Training
training_params = {
    "batch_size": "full",
    "max_epochs": 8000,
    "lr": 1e-6,
    "stop_patience": 1000,
    "stop_min_delta": 0.00,
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
def model_training(x, y, z, model_params, sim, progress):
    model = ClassGenCMI(x, y, z, **model_params)
    model.fit(**training_params)
    progress.update.remote()
    return (sim, model.get_cmi())


def read_data(key):
    data = np.load(f"{DATA_PATH}/{FILES[key]['data']}")
    true_cmi = np.load(f"{DATA_PATH}/{FILES[key]['ksg']}")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2:]

    return x, y, z, true_cmi[0]


def cmiTest01():
    print("Test 01/02")
    sims = FILES.keys()
    results = []
    sims_params = []
    # Simulations
    for sim in sims:
        # data
        x, y, z, true_cmi = read_data(sim)
        results += [(sim, "true value", true_cmi)]
        x_id = ray.put(x)
        y_id = ray.put(y)
        z_id = ray.put(z)
        gdim = 2 + z.shape[1]
        for _ in range(NREA):
            sim_params_ = {
                "x": x_id,
                "y": y_id,
                "z": z_id,
                "sim": sim,
                "model_params": {
                    "hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
                    "afn": afn,
                },
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
                s["model_params"],
                s["sim"],
                progress,
            )
            for s in sims_params
        ]
    )
    # parse-results
    results += [(sim_key, "ccmi-gen", cmi) for sim_key, cmi in res]
    results = pd.DataFrame(results, columns=["sim", "method", "cmi"])
    # plot
    sns.catplot(data=results, x="sim", y="cmi", hue="method", kind="bar")


def cmiTest02():
    print("Test 02/02")
    # data
    data_keys = FILES.keys()

    for key in data_keys:
        print(key)
        x, y, z, true_cmi = read_data(key)
        # model parameters
        gdim = 2 + z.shape[1]
        model_params = {
            "hidden_layers": [gdim / 2, gdim / 4, gdim / 8],
            "afn": afn,
        }

        class_cmigen_model = ClassGenCMI(x, y, z, **model_params)
        class_cmigen_model.fit(**training_params, verbose=True)
        cmi = class_cmigen_model.get_cmi()
        (
            val_cmi_epoch,
            val_loss_epoch,
        ) = class_cmigen_model.get_curves()

        fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
        epoch = np.arange(val_cmi_epoch.size)
        axs[0].set_title(f"CCMI-Gen sim {key}")
        axs[0].plot(epoch, val_cmi_epoch, "b", label="val cmi")
        axs[0].axhline(true_cmi, color="g", label="true value")
        axs[0].axhline(cmi, color="k", label="estimated cmi")
        axs[0].legend(loc="lower right")
        axs[1].plot(epoch, val_loss_epoch, "b", label="val loss")
        axs[1].legend(loc="upper right")
        axs[1].set_xlabel("Epoch", fontweight="bold")


if __name__ == "__main__":
    cmiTest01()
    cmiTest02()
    plt.show()

# def cmi(target, source, u, emb_params, model_params, train_params):
#     target = embedding(target, **emb_params)
#     source = embedding(source, **emb_params)
#
#     n = target.shape[0]
#     target_u = target[u:, :]
#     target = target[: n - u, :]
#     source = source[: n - u, :]
#     # I(target_u, source, target) - I(target_u,target)
#     class_cmi_model = ClassGenCMI(target_u, source, target, **model_params)
#     # training
#     class_cmi_model.fit(**train_params)
#     # Get mi estimation
#     cmi_test, cmi_val = class_cmi_model.get_cmi()
#     # Get curves
#     (
#         Dkl_val,
#         val_loss,
#         val_acc,
#     ) = class_cmi_model.get_curves()
#     return {
#         "cmi_test": cmi_test,
#         "cmi_val": cmi_val,
#         "Dkl_val_epoch": Dkl_val,
#         "val_loss_epoch": val_loss,
#         "val_acc_epoch": val_acc,
#     }
#
#
# def testClassCMI01():
#     emb_params = {"m": 1, "tau": 1}
#     u = 1
#     # model
#     model_params = {
#         "hidden_dim": 128,
#         "num_hidden_layers": 3,
#         "afn": "relu",
#     }
#     # embedding parameters
#     batch_size = 512
#     max_epochs = 8000
#     train_params = {
#         "batch_size": batch_size,
#         "max_epochs": max_epochs,
#         "knn": 2,
#         "lr": 1e-4,
#         "lr_factor": 0.5,
#         "lr_patience": 100,
#         "stop_patience": 300,
#         "stop_min_delta": 0.01,
#         "weight_decay": 1e-3,
#         "verbose": False,
#     }
#
#     n = 3000
#     C = np.linspace(0, 0.8, 13)
#     Txy = np.zeros_like(C)
#     Tyx = np.zeros_like(C)
#     for i, c in enumerate(tqdm(C)):
#         henon = coupledHenon(n, c)
#         X = np.squeeze(henon[:, 0])
#         Y = np.squeeze(henon[:, 1])
#         ret = cmi(Y, X, u, emb_params, model_params, train_params)
#         Txy[i] = ret["cmi_test"]
#         ret = cmi(X, Y, u, emb_params, model_params, train_params)
#         Tyx[i] = ret["cmi_test"]
#
#     fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
#     axs[0].set_title("Txy & Tyx")
#     axs[0].plot(C, Txy, "b", label="Txy")
#     axs[0].plot(C, Tyx, "r", label="Tyx")
#     axs[0].legend(loc="lower right")
#     axs[1].set_title("Txy - Tyx")
#     axs[1].plot(C, Txy - Tyx, "b")
#
#
# def testClassCMI02():
#     # Generate data
#     n = 3000
#     c = 0.45
#     henon = coupledHenon(n, c)
#     X = np.squeeze(henon[:, 0])
#     Y = np.squeeze(henon[:, 1])
#     emb_params = {"m": 1, "tau": 1}  # embedding parameters
#     u = 1
#     model_params = {  # model parameters
#         "hidden_dim": 128,
#         "num_hidden_layers": 3,
#         "afn": "relu",
#     }
#     batch_size = 256
#     max_epochs = 8000
#     train_params = {  # training parameters
#         "batch_size": batch_size,
#         "max_epochs": max_epochs,
#         "knn": 2,
#         "lr": 1e-4,
#         "lr_factor": 0.5,
#         "lr_patience": 100,
#         "stop_patience": 300,
#         "stop_min_delta": 0.01,
#         "weight_decay": 1e-3,
#         "verbose": True,
#     }
#
#     ret_xy = cmi(Y, X, u, emb_params, model_params, train_params)
#     ret_yx = cmi(X, Y, u, emb_params, model_params, train_params)
#
#     print(f"C={c}, X->Y CMI_TEST={ret_xy['cmi_test']}, CMI_VAL={ret_xy['cmi_val']}")
#     print(f"C={c}, Y->X CMI_TEST={ret_yx['cmi_test']}, CMI_VAL={ret_yx['cmi_val']}")
#
#     # Plot
#     fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
#     axs[0].set_title("Donsker-Varadhan representation")
#     axs[0].plot(ret_xy["Dkl_val_epoch"], "b", label="X -> Y")
#     axs[0].plot(ret_yx["Dkl_val_epoch"], "r", label="Y -> X")
#     axs[0].legend(loc="lower right")
#     axs[1].set_title("Accuracy")
#     axs[1].plot(ret_xy["val_acc_epoch"], "b", label="X -> Y")
#     axs[1].plot(ret_yx["val_acc_epoch"], "r", label="Y -> X")
#     axs[2].set_title("Cross-Entropy loss")
#     axs[2].plot(ret_xy["val_loss_epoch"], "b", label="X -> Y")
#     axs[2].plot(ret_yx["val_loss_epoch"], "r", label="Y -> X")
#     axs[3].set_title("CMI Txy - Tyx")
#     min_epoch = np.minimum(ret_xy["Dkl_val_epoch"].size, ret_yx["Dkl_val_epoch"].size)
#     axs[3].plot(
#         ret_xy["Dkl_val_epoch"][:min_epoch] - ret_yx["Dkl_val_epoch"][:min_epoch], "b"
#     )
#
#
# if __name__ == "__main__":
#     testClassCMI01()
#     testClassCMI02()
#     plt.show()
