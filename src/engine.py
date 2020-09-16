from argparse import ArgumentParser
import pytorch_lightning as pl

from .model import Model


class Engine(Model, pl.LightningModule):

    def __init__(self):
        super().__init__()

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument()

        return parser

    def configure_optimizers(self):
        optim = None
        scheduler = None
        return optim, scheduler

    def training_step(self, X):

        loss = None
        tb_log = {}
        logs = {'loss': loss, 'log': tb_log}
        return logs

    def validation_step(self, X):

        loss = None
        tb_log = {}
        logs = {'val_loss': loss, 'log': tb_log}
        return logs

    def test_step(self, X):

        loss = None
        tb_log = {}
        logs = {'test_loss': loss, 'log': tb_log}
        return logs
