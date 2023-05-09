#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from lightning import LightningModule, Trainer
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
        logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
        logging.WARNING)
    st = 'auto'
except ImportError:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    st = None

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

from minepy.mineLayers import get_activation_fn
from minepy.mineTools import toColVector

logging.basicConfig()
logger = logging.getLogger(__name__)

EPS = 1e-6


class EMALoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema
    return t_log, running_mean


class MineNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, afn, nLayers, loss, alpha,
                 regWeight, targetVal):
        super().__init__()
        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(nLayers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*seq)
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.regWeight = regWeight
        self.targetVal = targetVal

    def forward(self, x, z):
        z_marg = z[torch.randperm(z.shape[0])]

        t = self.net(torch.cat((x, z), dim=1)).mean()
        t_marg = self.net(torch.cat((x, z_marg), dim=1))
        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg,
                                                      self.running_mean,
                                                      self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ["remine"]:
            second_term, self.running_mean = ema_loss(t_marg,
                                                      self.running_mean,
                                                      self.alpha)
            second_term += self.regWeight * torch.pow(
                second_term - self.targetVal, 2)
        else:
            second_term = torch.logsumexp(t_marg, 0) - math.log(
                t_marg.shape[0])  # mine_biased as default

        return -t + second_term


class Mine(LightningModule):

    def __init__(
        self,
        input_dim,
        hidden_dim=150,
        afn="elu",
        nLayers=3,
        loss='mine_biased',
        alpha=0.01,
        regWeight=1.0,
        targetVal=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = MineNet(input_dim, hidden_dim, afn, nLayers, loss, alpha,
                           regWeight, targetVal)
        self.ema_val_loss = None
        self.calc_curves = False
        self.val_loss_epoch = []
        self.test_loss_epoch = []
        self.ema_val_loss_epoch = []

    def loss(self, x, z):
        return self.net.forward(x, z)

    def training_step(self, batch, batch_idx):
        x, z = batch
        loss = self.loss(x, z)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z = batch
        val_loss = self.loss(x, z).detach()
        # calculate ema_val_loss
        if self.ema_val_loss is None:
            self.ema_val_loss = val_loss
        else:
            self.ema_val_loss = ema(val_loss, 0.01, self.ema_val_loss)

        self.val_loss_epoch.append(val_loss.item())
        self.ema_val_loss_epoch.append(self.ema_val_loss.item())
        # test loss
        test_loss = self.loss(self.x, self.z)
        self.test_loss_epoch.append(test_loss.item())
        self.log_dict({'val_loss': self.ema_val_loss, 'test_loss': test_loss})
        # return {'val_loss': self.ema_val_loss, 'test_loss': test_loss}
        # return {'val_loss': self.ema_val_loss, 'test_loss': test_loss}

    def get_mi(self):
        self.calc_loss_curves()
        ind_max_stop = np.argmax(self.ema_val_loss_epoch)
        ind_max_val = np.argmax(self.val_loss_epoch)
        mi_val = self.val_loss_epoch[ind_max_val]
        mi_test = self.test_loss_epoch[ind_max_val]
        mi_stop = self.test_loss_epoch[ind_max_stop]
        fepoch = self.val_loss_epoch.size
        return (mi_val, mi_test, mi_stop, ind_max_val, ind_max_stop, fepoch)

    def calc_loss_curves(self):
        if self.calc_curves:
            pass
        else:
            self.val_loss_epoch = -np.array(self.val_loss_epoch)
            self.test_loss_epoch = -np.array(self.test_loss_epoch)
            self.ema_val_loss_epoch = -np.array(self.ema_val_loss_epoch)
            self.calc_curves = True
        return self.val_loss_epoch, self.test_loss_epoch, self.ema_val_loss_epoch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=self.lr_factor,
                                      patience=self.lr_patience,
                                      verbose=self.verbose)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            # "monitor": "train_loss"
            "monitor": "val_loss"
        }

    def fit(self,
            X,
            Z,
            batch_size=64,
            max_epochs=1000,
            val_size=0.2,
            lr=1e-3,
            lr_patience=100,
            lr_factor=0.25,
            weight_decay=5e-5,
            early_st_patience=10,
            early_st_min_delta=0,
            callbacks=None,
            device=None,
            trainer_devices='auto',
            num_nodes=1,
            strategy=None,
            verbose=False):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        accelerator = 'gpu' if device == 'cuda' else 'cpu'
        strategy = st if strategy is None else strategy

        callbacks = callbacks or []
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        X = torch.from_numpy(toColVector(X.astype(np.float32)))
        Z = torch.from_numpy(toColVector(Z.astype(np.float32)))
        # Z = torch.tensor(toColVector(Z))
        self.x = X.to(device)
        self.z = Z.to(device)

        N, _ = X.shape

        valid_size = int(val_size * N)
        train_size = int(N - valid_size)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train, valid = random_split(TensorDataset(X, Z),
                                        lengths=[train_size, valid_size])
            train_dataloader = DataLoader(train,
                                          batch_size=batch_size,
                                          shuffle=True)
            val_dataloader = DataLoader(valid, batch_size=valid_size)
            early_stopping = EarlyStopping(mode='min',
                                           monitor='val_loss',
                                           min_delta=early_st_min_delta,
                                           patience=early_st_patience,
                                           verbose=self.verbose)
            callbacks.append(early_stopping)
            trainer = Trainer(accelerator=accelerator,
                              devices=trainer_devices,
                              num_nodes=num_nodes,
                              strategy=strategy,
                              max_epochs=max_epochs,
                              callbacks=callbacks,
                              deterministic=False,
                              val_check_interval=1.0,
                              logger=self.verbose,
                              enable_checkpointing=self.verbose,
                              enable_progress_bar=self.verbose,
                              enable_model_summary=self.verbose,
                              detect_anomaly=self.verbose)
            trainer.fit(model=self,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)
            logs, = trainer.validate(model=self,
                                     dataloaders=val_dataloader,
                                     verbose=self.verbose)

        return logs
