import numpy as np
import torch
from scipy.stats import norm
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from tqdm import tqdm

from minepy.dine.dine_tools import UniformFlow, data_loader
from minepy.minepy_tools import (EarlyStopping, ExpMovingAverageSmooth,
                                 toColVector)

EPS = 1e-6


class DineCMI(nn.Module):
    def __init__(
        self,
        X,
        Y,
        Z,
        n_components,
        hidden_sizes,
        device=None,
    ):
        super().__init__()
        # select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)
        # Vars
        self.X = torch.from_numpy(toColVector(X.astype(np.float32))).to(self.device)
        self.Y = torch.from_numpy(toColVector(Y.astype(np.float32))).to(self.device)
        self.Z = torch.from_numpy(toColVector(Z.astype(np.float32))).to(self.device)
        dx = self.X.shape[1]
        dy = self.Y.shape[1]
        dz = self.Z.shape[1]

        self.x_cnf = UniformFlow(
            d=dx, dz=dz, n_components=n_components, dh=hidden_sizes
        ).to(self.device)
        self.y_cnf = UniformFlow(
            d=dy, dz=dz, n_components=n_components, dh=hidden_sizes
        ).to(self.device)

    def loss(self, X, Y, Z):
        _, log_px = self.x_cnf(X, Z)
        _, log_py = self.y_cnf(Y, Z)

        loss = -torch.mean(log_px + log_py)
        return loss

    def transform(self, X, Y, Z):
        self.eval()
        ex, _ = self.x_cnf(X, Z)
        ey, _ = self.y_cnf(Y, Z)

        ex = ex.detach().cpu().numpy()
        ey = ey.detach().cpu().numpy()
        return ex, ey

    def fit(
        self,
        batch_size=64,
        max_epochs=2000,
        lr=1e-4,
        weight_decay=5e-5,
        val_size=0.2,
        verbose=False,
    ):
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            opt, factor=0.5, patience=100, mode="min", verbose=verbose
        )
        # opt = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = CyclicLR(opt, base_lr=lr, max_lr=1e-5, mode="triangular2")

        Xtrain, Ytrain, Ztrain, Xval, Yval, Zval, X, Y, Z = data_loader(
            self.X, self.Y, self.Z, val_size=val_size, device=self.device
        )

        val_loss_epoch = []
        cmi_epoch = []

        # training
        self.train()
        for _ in tqdm(range(max_epochs), disable=not verbose):
            n = Xtrain.shape[0]
            rand_perm = torch.randperm(n)
            if batch_size == "full":
                batch_size = n
            self.train()
            for inds in rand_perm.split(batch_size, dim=0):
                with torch.set_grad_enabled(True):
                    opt.zero_grad()
                    train_loss = self.loss(
                        Xtrain[inds, :], Ytrain[inds, :], Ztrain[inds, :]
                    )
                    train_loss.backward()
                    opt.step()

            self.eval()
            with torch.set_grad_enabled(False):
                val_loss = self.loss(Xval, Yval, Zval)
                val_loss_epoch.append(val_loss.item())
                cmi_epoch.append(self.get_cmi())
                # learning rate scheduler
                scheduler.step(val_loss)
                # scheduler.step()
        self.val_loss_epoch = np.array(val_loss_epoch)
        self.cmi_epoch = np.array(cmi_epoch)

    def get_cmi(self):
        e_x, e_y = self.transform(self.X, self.Y, self.Z)
        e_x, e_y = map(lambda x: np.clip(x, EPS, 1 - EPS), (e_x, e_y))
        e_x, e_y = map(norm.ppf, (e_x, e_y))
        cov_x, cov_y = map(np.cov, (e_x.T, e_y.T))
        cov_x = cov_x.reshape(e_x.shape[1], e_x.shape[1])
        cov_y = cov_y.reshape(e_y.shape[1], e_y.shape[1])
        cov_all = np.cov(np.column_stack((e_x, e_y)).T)
        mi = 0.5 * (
            np.log(np.linalg.det(cov_x))
            + np.log(np.linalg.det(cov_y))
            - np.log(np.linalg.det(cov_all))
        )
        return mi

    def get_curves(self):
        return self.val_loss_epoch, self.cmi_epoch
