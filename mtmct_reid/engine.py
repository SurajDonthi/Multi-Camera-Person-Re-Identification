from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import lr_scheduler
from torch.optim.sgd import SGD
from typing import Union, List

from metrics import joint_scores, mAP
from model import PCB
from re_ranking import re_ranking
from utils import fliplr, l2_norm_standardize, plot_distributions

# class ST_ReIDMeta(type(ReIDDataLoader), type(pl.LightningModule), type(PCB)):
#     pass

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}


class ST_ReID(PCB, pl.LightningModule):

    def __init__(self, num_classes, learning_rate: float = 0.1,
                 criterion: str = 'cross_entropy', rerank: bool = False):

        super().__init__(num_classes)

        self.learning_rate = learning_rate
        self.criterion = LOSSES[criterion]
        self.rerank = rerank

        self.save_hyperparameters()

    @ staticmethod   # @classmethod not required as
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-lp', '--log_path', type=str,
                            default='./lightning_logs')
        parser.add_argument('-lr', '--learning_rate',
                            type=float, default=0.1)
        parser.add_argument('-c', '--criterion', type=str,
                            choices=LOSSES.keys(),
                            default='cross_entropy')
        parser.add_argument('-re', '--rerank',
                            type=bool, default=False)
        return parser

    def configure_optimizers(self):
        # Add args to define the values
        params_list = [{'params': self.model.parameters(), 'lr': 0.01},
                       {'params': self.classifier0.parameters()},
                       {'params': self.classifier1.parameters()},
                       {'params': self.classifier2.parameters()},
                       {'params': self.classifier3.parameters()},
                       {'params': self.classifier4.parameters()},
                       {'params': self.classifier5.parameters()}]

        optim = SGD(params_list,
                    lr=self.learning_rate,
                    weight_decay=5e-4, momentum=0.9, nesterov=True)

        scheduler = lr_scheduler.StepLR(optim, step_size=40, gamma=0.130)

        return [optim], [scheduler]

    def shared_step(self, batch, batch_idx):
        X, y, _, _ = batch
        parts_proba = self(X)    # returns a list of parts

        loss = 0
        y_pred = torch.zeros_like(parts_proba[0])
        for part in parts_proba:
            loss += self.criterion(part, y)
            y_pred += F.softmax(part)

        y_pred = torch.stack(parts_proba, dim=0).sum(dim=0)
        _, y_pred = torch.max(y_pred, dim=1)
        acc = accuracy(y_pred, y)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)

        result = pl.TrainResult(loss)
        logs = {'Loss/train_loss': loss, 'Accuracy/train_acc': acc}
        result.log_dict(logs, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)

        result = pl.EvalResult()
        logs = {'Loss/val_loss': loss, 'Accuracy/val_acc': acc}
        result.log_dict(logs, prog_bar=True)
        return result

    def on_test_epoch_start(self) -> None:
        # self.all_features = torch.Tensor()
        self.q_features = torch.Tensor()
        self.q_targets = torch.Tensor()
        self.q_cam_ids = torch.Tensor()
        self.q_frames = torch.Tensor()
        self.g_features = torch.Tensor()
        self.g_targets = torch.Tensor()
        self.g_cam_ids = torch.Tensor()
        self.g_frames = torch.Tensor()

    def test_step(self, batch, batch_idx, dataloader_idx):
        X, y, cam_ids, frames = batch

        feature_sum = 0
        for _ in range(2):
            features = self(X, training=False).detach().cpu()
            feature_sum += features
            X = fliplr(X, self.device)

        if dataloader_idx == 0:

            self.q_features = torch.cat([self.q_features, feature_sum])
            self.q_targets = torch.cat(
                [self.q_targets, y.detach().cpu()])
            self.q_cam_ids = torch.cat(
                [self.q_cam_ids, cam_ids.detach().cpu()])
            self.q_frames = torch.cat(
                [self.q_frames, frames.detach().cpu()])

        elif dataloader_idx == 1:

            self.g_features = torch.cat([self.g_features, feature_sum])
            self.g_targets = torch.cat(
                [self.g_targets, y.detach().cpu()])
            self.g_cam_ids = torch.cat(
                [self.g_cam_ids, cam_ids.detach().cpu()])
            self.g_frames = torch.cat(
                [self.g_frames, frames.detach().cpu()])

        result = pl.EvalResult()

        loss, acc = self.shared_step(batch, batch_idx)

        logs = {'Loss/test_loss': loss, 'Accuracy/test_acc': acc}
        result.log_dict(logs, prog_bar=True)

        return result

    def test_epoch_end(self,
                       outputs: Union[pl.EvalResult, List[pl.EvalResult]]) \
            -> pl.EvalResult:

        fig = plot_distributions(self.trainer.datamodule.st_distribution)

        self.logger[0].experiment.add_figure('Spatial-Temporal Distribution',
                                             fig)

        # ? Find out what `outputs` is.
        self.q_features = l2_norm_standardize(self.q_features)
        self.g_features = l2_norm_standardize(self.g_features)

        scores = joint_scores(self.q_features,
                              self.q_cam_ids,
                              self.q_frames,
                              self.g_features,
                              self.g_cam_ids,
                              self.g_frames,
                              self.trainer.datamodule.st_distribution)

        if self.rerank:
            scores = re_ranking(scores)

        mean_ap, cmc = mAP(scores, self.q_targets,
                           self.q_cam_ids,
                           self.g_targets,
                           self.g_cam_ids)

        mAP_logs = {'Results/mAP': mean_ap, 'Results/CMC_top1': cmc[0].tolist(),
                    'Results/CMC_top5': cmc[4].tolist()}

        loss = torch.mean(outputs[1]['Loss/test_loss'])
        acc = torch.mean(outputs[1]['Accuracy/test_acc'])

        log = {'Results/loss': loss.tolist(), 'Results/accuracy': acc.tolist(),
               **mAP_logs}

        out = {**log, **mAP_logs}

        result = pl.EvalResult()
        result.log_dict(out)
        return result
