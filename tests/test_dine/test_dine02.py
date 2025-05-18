#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ray
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from minepy.dine.dine import Dine
from minepy_tests.testTools import (FILES, Progress, cmi_non_lin_samples01,
                                    cmi_non_lin_samples02, read_data)

TEST_FILES = [
    "lf_5kdz20",
    "lf_10kdz20",
    "nl_5kdz10",
    "nl_10kdz20",
]

NREA = 5  # number of realizations
MAX_ACTORS = 5  # number of nodes

# model parameters
model_params = {"n_components": 64, "hidden_sizes": 8}
# training parameters
training_params = {
    "batch_size": 256,
    "max_epochs": 3000,
    "lr": 1e-4,
    "stop_patience": 200,
    "stop_min_delta": 0,
    "val_size": 0.2,
}


# @ray.remote(num_cpus= MAX_ACTORS)
@ray.remote(num_gpus=1 / MAX_ACTORS, max_calls=1)
def model_training(x, y, z, model_params, sim, progress):
    model = Dine(x, y, z, **model_params)
    model.fit(**training_params)
    progress.update.remote()
    return (sim, model.get_cmi())


def cmiTest01():
    print("Test 01/02")
    results = []
    sims_params = []
    # Simulations
    for sim in TEST_FILES:
        # data
        x, y, z, true_cmi = read_data(sim)
        results += [(sim, "true value", true_cmi)]
        x_id = ray.put(x)
        y_id = ray.put(y)
        z_id = ray.put(z)
        for _ in range(NREA):
            sim_params_ = {
                "x": x_id,
                "y": y_id,
                "z": z_id,
                "sim": sim,
                "model_params": model_params,
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
    results += [(sim_key, "dine", cmi) for sim_key, cmi in res]
    results = pd.DataFrame(results, columns=["sim", "method", "cmi"])
    # plot
    error = (
        results.groupby(["sim", "method"])
        .apply(lambda x: x["cmi"].std())
        .reset_index(name="error")
    )
    fig = go.Figure()
    x = results["sim"].unique()
    y = results[results["method"] == "dine"]["cmi"]
    err = error[error["method"] == "dine"]["error"]
    fig.add_trace(go.Bar(name="DINE", x=x, y=y, error_y=dict(type="data", array=err)))
    x = results["sim"].unique()
    y = results[results["method"] == "true value"]["cmi"]
    err = error[error["method"] == "true value"]["error"]
    fig.add_trace(
        go.Bar(name="True value", x=x, y=y, error_y=dict(type="data", array=err))
    )
    fig.show()


def cmiTest02():
    print("Test 02/02")
    N = 3000
    sims = [TEST_FILES[2]]
    # RHO = [0.9]
    # Simulations
    for sim in sims:
        # for rho in RHO:
        # data
        x, y, z, true_cmi = read_data(sim)
        # x, y, z, true_cmi, est_cmi = cmi_non_lin_samples01(
        #     N, 4, rho=rho, random_state=3
        # )
        # x, y, z = cmi_non_lin_samples02(N, 4, random_state=2)
        # true_cmi = 0

        dine_model = Dine(x, y, z, **model_params)
        dine_model.fit(**training_params, verbose=True)
        cmi = dine_model.get_cmi()
        val_loss_epoch, cmi_epoch = dine_model.get_curves()

        print(sim + ":")
        print(f"true={true_cmi}, estimated={cmi}")
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(
            go.Scatter(
                x=np.arange(val_loss_epoch.size), y=val_loss_epoch, name="val loss"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=np.arange(cmi_epoch.size), y=cmi_epoch, name="cmi"),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=true_cmi,
            name="true value",
            line_dash="dot",
            row=2,
            col=1,
        )
        fig.update_layout(title_text=f"Dine - sim: {sim}")
        fig.show()


if __name__ == "__main__":
    # cmiTest01()
    cmiTest02()
