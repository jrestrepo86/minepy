#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
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

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

from minepy.mine.mine_tools import MineModel, ema
from minepy.minepy_tools import toColVector

logging.basicConfig()
logger = logging.getLogger(__name__)


class Mine(LightningModule):

    def __init__(
        self,
        input_dim,
        hidden_dim=50,
        num_hidden_layers=2,
        afn="elu",
        loss='mine_biased',
        alpha=0.1,
        regWeight=1,
        targetVal=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MineModel(input_dim, hidden_dim, afn, num_hidden_layers,
                               loss, alpha, regWeight, targetVal)
        self.calc_curves = False

        self.val_ema_loss = None
        self.train_loss_epoch = []
        self.train_mi_epoch = []
        self.val_loss_epoch = []
        self.val_ema_loss_epoch = []
        self.val_mi_epoch = []
        self.test_loss_epoch = []
        self.test_mi_epoch = []

    def loss(self, x, z):
        loss, mi = self.model.forward(x, z)
        return loss, mi

    def training_step(self, batch, batch_idx):
        x, z = batch
        loss, mi = self.loss(x, z)
        self.train_loss_epoch.append(loss.detach().item())
        self.train_mi_epoch.append(mi.detach().item())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, z = batch
        val_loss, val_mi = self.loss(x, z)
        val_loss = val_loss.detach()
        val_mi = val_mi.detach()
        # calculate validation ema loss
        if self.val_ema_loss is None:
            self.val_ema_loss = val_loss
        else:
            self.val_ema_loss = ema(val_loss, 0.01, self.val_ema_loss)

        self.val_mi_epoch.append(val_mi.item())
        self.val_ema_loss_epoch.append(self.val_ema_loss.item())
        # test loss
        _, test_mi = self.loss(self.x, self.z)
        self.test_mi_epoch.append(test_mi.item())
        self.log('val_loss', self.val_ema_loss)
        return self.val_ema_loss
        # return {'val_loss': self.ema_val_loss, 'test_loss': test_loss}
        # return {'val_loss': self.ema_val_loss, 'test_loss': test_loss}

    def get_mi(self, all=False):
        self.train_loss_epoch = np.array(self.train_loss_epoch)
        self.train_mi_epoch = np.array(self.train_mi_epoch)
        self.val_loss_epoch = np.array(self.val_loss_epoch)
        self.val_mi_epoch = np.array(self.val_mi_epoch)
        self.test_mi_epoch = np.array(self.test_mi_epoch)
        self.val_ema_loss_epoch = np.array(self.val_ema_loss_epoch)
        self.val_ema_mi_epoch = -self.val_ema_loss_epoch

        ind_max_stop = np.argmax(self.val_ema_mi_epoch)
        ind_min_stop = np.argmin(self.val_ema_loss_epoch)
        ind_max_val = np.argmax(self.val_mi_epoch)
        mi_val = self.val_mi_epoch[ind_max_val]
        mi_test = self.test_mi_epoch[ind_max_val]
        mi_stop = self.test_mi_epoch[ind_max_stop]
        fepoch = self.val_mi_epoch.size
        if all:
            return (mi_val, mi_test, mi_stop, ind_max_val, ind_min_stop,
                    fepoch)
        else:
            return mi_test

    def calc_loss_curves(self):
        _ = self.get_mi()
        return (self.train_loss_epoch, self.val_loss_epoch, self.test_mi_epoch,
                self.val_ema_loss_epoch)

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
            lr_factor=0.1,
            lr_patience=10,
            weight_decay=5e-5,
            stop_patience=100,
            stop_min_delta=0.05,
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
                                           min_delta=stop_min_delta,
                                           patience=stop_patience,
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
