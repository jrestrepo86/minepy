import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F

from minepy.minepy_tools import get_activation_fn

EPS = 1e-6


def MLP(
    in_features,
    out_features,
    hidden_sizes,
    afn="relu",
):
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = [in_features] + hidden_sizes + [out_features]

    activation_fn = get_activation_fn(afn)
    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if i < len(hidden_sizes) - 2:
            layers.append(activation_fn())
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))

    return nn.Sequential(*layers)


class UniformFlow(nn.Module):
    def __init__(
        self,
        dz,
        n_components,
        hidden_sizes,
        afn="leaky_relu",
    ):
        super().__init__()
        self.z_to_mu = MLP(dz, n_components, hidden_sizes=hidden_sizes, afn=afn)
        self.z_to_logstd = MLP(dz, n_components, hidden_sizes=hidden_sizes, afn=afn)
        self.z_to_w = nn.Sequential(
            *MLP(dz, n_components, hidden_sizes=hidden_sizes, afn=afn),
            nn.Softmax(dim=1),
        )

    def forward(self, X, Z):
        # X = X.view(X.shape[0], -1)  # N x 1
        mu = self.z_to_mu(Z)  # N x k
        std = self.z_to_logstd(Z).exp()  # N x k
        std = torch.clip(std, min=1e-6, max=None)
        w = self.z_to_w(Z)  # N x k

        dist = Normal(mu, std)  # N x k
        e = (dist.cdf(X) * w).sum(axis=1, keepdims=True)
        p_hat = (dist.log_prob(X).exp() * w).sum(axis=1, keepdims=True)
        p_hat = torch.clip(p_hat, min=1e-24, max=None)
        log_de_dx = p_hat.log()

        return e, log_de_dx


def data_loader(X, Y, Z, val_size=0.2, device="cuda"):
    n = X.shape[0]

    # mix samples
    # inds = np.random.permutation(n)
    # X = X[inds, :].to(device)
    # Y = Y[inds, :].to(device)
    # Z = Z[inds, :].to(device)
    # split data in training and validation sets
    val_size = int(val_size * n)
    inds = torch.randperm(n)
    (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

    Xtrain = X[train_idx, :]
    Ytrain = Y[train_idx, :]
    Ztrain = Z[train_idx, :]
    Xval = X[val_idx, :]
    Yval = Y[val_idx, :]
    Zval = Z[val_idx, :]
    Xtrain = Xtrain.to(device)
    Ytrain = Ytrain.to(device)
    Ztrain = Ztrain.to(device)
    Xval = Xval.to(device)
    Yval = Yval.to(device)
    Zval = Zval.to(device)
    return Xtrain, Ytrain, Ztrain, Xval, Yval, Zval, X, Y, Z
