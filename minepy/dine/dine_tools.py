from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal
from torch.nn import functional as F

EPS = 1e-6


def MLP(d_in: int, d_out: int, hidden_sizes: List[int] = None):
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = [d_in] + (hidden_sizes or []) + [d_out]

    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if i < len(hidden_sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))

    return nn.Sequential(*layers)


class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, mask: Tensor, bias: bool = True) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(n_in, n_out, bias)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask.T * self.weight, self.bias)


def get_mask(d_in, d_out, n_groups, mask_type, is_output):
    g_in, g_out = d_in // n_groups, d_out // n_groups
    x, y = torch.meshgrid(torch.arange(d_in), torch.arange(d_out), indexing="ij")
    x = torch.div(x, g_in, rounding_mode="floor")
    y = torch.div(y, g_out, rounding_mode="floor")
    if mask_type == "autoregressive":
        if is_output:
            mask = (x < y).float()
        else:
            mask = (x <= y).float()
    elif mask_type == "grouped":
        mask = (x == y).float()
    return mask


def MaskedMLP(d_in, d_out, n_groups, hidden_sizes, mask_type):
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = [d_in] + hidden_sizes + [d_out]
    act_func = lambda: nn.ReLU(inplace=True)
    layers = []
    for i in range(len(hidden_sizes) - 1):
        mask = get_mask(
            hidden_sizes[i],
            hidden_sizes[i + 1],
            n_groups,
            mask_type,
            is_output=(i == len(hidden_sizes) - 2),
        )
        layers.append(MaskedLinear(hidden_sizes[i], hidden_sizes[i + 1], mask=mask))
        if i < len(hidden_sizes) - 2:
            layers.append(act_func())
        # layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))

    return nn.Sequential(*layers)


class UniformFlow(nn.Module):
    def __init__(self, d, dz, dh, n_components):
        super().__init__()
        self.Z_encoder = MLP(d_in=dz, d_out=dh, hidden_sizes=dh)
        self.conditioner = MaskedMLP(
            d_in=d, d_out=d * dh, n_groups=d, hidden_sizes=d, mask_type="autoregressive"
        )
        self.c_to_gmm = MaskedMLP(
            d_in=d * dh,
            d_out=d * n_components * 3,
            n_groups=d,
            hidden_sizes=d * dh,
            mask_type="grouped",
        )

        self.d = d
        self.dz = dz
        self.dh = dh
        self.n_components = n_components

    def forward(self, X, Z):
        N = X.shape[0]

        c = self.conditioner(X) + self.Z_encoder(Z).repeat(1, self.d)  # N x [d x h]
        gmm = self.c_to_gmm(c).view(N, self.d, -1)
        mu = gmm[..., : self.n_components]
        std = gmm[..., self.n_components : 2 * self.n_components].exp()
        std = torch.clip(std, min=1e-6, max=None)
        logits = gmm[..., -self.n_components :]
        w = torch.softmax(logits, dim=2)  # N x d x k

        dist = Normal(mu, std)  # N x d x k
        e = (dist.cdf(X[..., None]) * w).sum(dim=2)  # N x d
        p_hat = (dist.log_prob(X[..., None]).exp() * w).sum(dim=2)
        p_hat = torch.clip(p_hat, min=1e-24, max=None)
        # print(p_hat.min(), (1 / p_hat).max())
        log_de_dx = p_hat.log().sum(axis=1)  # N

        assert not torch.isnan(e).any()
        assert not torch.isnan(log_de_dx).any()
        # assert log_de_dx.isnan().any() == False

        return e, log_de_dx


def data_loader(X, Y, Z, val_size=0.2, device="cuda"):
    n = X.shape[0]
    # split data in training and validation sets
    val_size = int(val_size * n)
    inds = torch.randperm(n)
    (val_idx, train_idx) = (inds[:val_size], inds[val_size:])

    Xtrain = X[train_idx, :].to(device)
    Ytrain = Y[train_idx, :].to(device)
    Ztrain = Z[train_idx, :].to(device)
    Xval = X[val_idx, :].to(device)
    Yval = Y[val_idx, :].to(device)
    Zval = Z[val_idx, :].to(device)
    return Xtrain, Ytrain, Ztrain, Xval, Yval, Zval
