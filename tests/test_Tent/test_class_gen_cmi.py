#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from minepy.class_mi.class_gen_cmi import ClassGenCMI
from minepy.minepy_tools import coupledHenon, embedding


def cmi(target, source, u, emb_params, model_params, train_params):
    target = embedding(target, **emb_params)
    source = embedding(source, **emb_params)

    n = target.shape[0]
    target_u = target[u:, :]
    target = target[: n - u, :]
    source = source[: n - u, :]
    # I(target_u, source, target) - I(target_u,target)
    class_cmi_model = ClassGenCMI(target_u, source, target, **model_params)
    # training
    class_cmi_model.fit(**train_params)
    # Get mi estimation
    cmi_test, cmi_val = class_cmi_model.get_cmi()
    # Get curves
    (
        Dkl_val,
        val_loss,
        val_acc,
    ) = class_cmi_model.get_curves()
    return {
        "cmi_test": cmi_test,
        "cmi_val": cmi_val,
        "Dkl_val_epoch": Dkl_val,
        "val_loss_epoch": val_loss,
        "val_acc_epoch": val_acc,
    }


def testClassCMI01():
    emb_params = {"m": 1, "tau": 1}
    u = 1
    # model
    model_params = {
        "hidden_dim": 128,
        "num_hidden_layers": 3,
        "afn": "relu",
    }
    # embedding parameters
    batch_size = 512
    max_epochs = 8000
    train_params = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "knn": 2,
        "lr": 1e-4,
        "lr_factor": 0.5,
        "lr_patience": 100,
        "stop_patience": 300,
        "stop_min_delta": 0.01,
        "weight_decay": 1e-3,
        "verbose": False,
    }

    n = 3000
    C = np.linspace(0, 0.8, 13)
    Txy = np.zeros_like(C)
    Tyx = np.zeros_like(C)
    for i, c in enumerate(tqdm(C)):
        henon = coupledHenon(n, c)
        X = np.squeeze(henon[:, 0])
        Y = np.squeeze(henon[:, 1])
        ret = cmi(Y, X, u, emb_params, model_params, train_params)
        Txy[i] = ret["cmi_test"]
        ret = cmi(X, Y, u, emb_params, model_params, train_params)
        Tyx[i] = ret["cmi_test"]

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    axs[0].set_title("Txy & Tyx")
    axs[0].plot(C, Txy, "b", label="Txy")
    axs[0].plot(C, Tyx, "r", label="Tyx")
    axs[0].legend(loc="lower right")
    axs[1].set_title("Txy - Tyx")
    axs[1].plot(C, Txy - Tyx, "b")


def testClassCMI02():
    # Generate data
    n = 3000
    c = 0.45
    henon = coupledHenon(n, c)
    X = np.squeeze(henon[:, 0])
    Y = np.squeeze(henon[:, 1])
    emb_params = {"m": 1, "tau": 1}  # embedding parameters
    u = 1
    model_params = {  # model parameters
        "hidden_dim": 128,
        "num_hidden_layers": 3,
        "afn": "relu",
    }
    batch_size = 256
    max_epochs = 8000
    train_params = {  # training parameters
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "knn": 2,
        "lr": 1e-4,
        "lr_factor": 0.5,
        "lr_patience": 100,
        "stop_patience": 300,
        "stop_min_delta": 0.01,
        "weight_decay": 1e-3,
        "verbose": True,
    }

    ret_xy = cmi(Y, X, u, emb_params, model_params, train_params)
    ret_yx = cmi(X, Y, u, emb_params, model_params, train_params)

    print(f"C={c}, X->Y CMI_TEST={ret_xy['cmi_test']}, CMI_VAL={ret_xy['cmi_val']}")
    print(f"C={c}, Y->X CMI_TEST={ret_yx['cmi_test']}, CMI_VAL={ret_yx['cmi_val']}")

    # Plot
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
    axs[0].set_title("Donsker-Varadhan representation")
    axs[0].plot(ret_xy["Dkl_val_epoch"], "b", label="X -> Y")
    axs[0].plot(ret_yx["Dkl_val_epoch"], "r", label="Y -> X")
    axs[0].legend(loc="lower right")
    axs[1].set_title("Accuracy")
    axs[1].plot(ret_xy["val_acc_epoch"], "b", label="X -> Y")
    axs[1].plot(ret_yx["val_acc_epoch"], "r", label="Y -> X")
    axs[2].set_title("Cross-Entropy loss")
    axs[2].plot(ret_xy["val_loss_epoch"], "b", label="X -> Y")
    axs[2].plot(ret_yx["val_loss_epoch"], "r", label="Y -> X")
    axs[3].set_title("CMI Txy - Tyx")
    min_epoch = np.minimum(ret_xy["Dkl_val_epoch"].size, ret_yx["Dkl_val_epoch"].size)
    axs[3].plot(
        ret_xy["Dkl_val_epoch"][:min_epoch] - ret_yx["Dkl_val_epoch"][:min_epoch], "b"
    )


if __name__ == "__main__":
    testClassCMI01()
    testClassCMI02()
    plt.show()
