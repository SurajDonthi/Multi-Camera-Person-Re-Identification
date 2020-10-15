from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch.optim import lr_scheduler
from torch.optim.sgd import SGD
from typing import Any, List

from metrics import joint_scores, mAP
from model import PCB
from re_ranking import re_ranking
from utils import fliplr, l2_norm_standardize, plot_distributions


LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}


class ST_ReID(PCB, pl.LightningModule):

    def __init__(self, num_classes, learning_rate: float = 0.1,
                 criterion: str = 'cross_entropy', rerank: bool = False):

        super().__init__(num_classes)

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.criterion = LOSSES[criterion]
        self.rerank = rerank

        # self.save_hyperparameters()

    @staticmethod   # @classmethod not required as
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

    # Shared steps
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

    def eval_shared_step(self, batch, batch_idx, dataloader_idx):
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

        loss, acc = self.shared_step(batch, batch_idx)

        return loss, acc

    def evaluation_metrics(self, outputs):
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

        avg_loss = torch.mean(outputs[0])
        avg_acc = torch.mean(outputs[1])

        return avg_loss, avg_acc, mean_ap, cmc

    # Training
    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx)

        logs = {'Loss/train_loss': loss, 'Accuracy/train_acc': acc}
        self.log_dict(logs, prog_bar=True)
        return loss, acc

    # Validation
    def on_validation_epoch_start(self) -> None:
        # self.all_features = torch.Tensor()
        self.q_features = torch.Tensor()
        self.q_targets = torch.Tensor()
        self.q_cam_ids = torch.Tensor()
        self.q_frames = torch.Tensor()
        self.g_features = torch.Tensor()
        self.g_targets = torch.Tensor()
        self.g_cam_ids = torch.Tensor()
        self.g_frames = torch.Tensor()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, acc = self.eval_shared_step(batch, batch_idx, dataloader_idx)

        logs = {'Loss/val_loss': loss, 'Accuracy/val_acc': acc}
        result = pl.EvalResult(early_stop_on=loss)
        result.log_dict(logs, prog_bar=True)
        return result

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss, avg_acc, mean_ap, cmc = self.evaluation_metrics(outputs)

        mAP_logs = {'Results/val_mAP': mean_ap,
                    'Results/val_CMC_top1': cmc[0].tolist(),
                    'Results/val_CMC_top5': cmc[4].tolist()}

        log = {'Results/val_loss': avg_loss.tolist(),
               'Results/val_accuracy': avg_acc.tolist(),
               **mAP_logs}

        out = {**log, **mAP_logs, 'step': self.current_epoch}
        result = pl.EvalResult()
        result.log_dict(out)
        return result

    # Testing
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
        loss, acc = self.eval_shared_step(batch, batch_idx, dataloader_idx)

        logs = {'Loss/test_loss': loss, 'Accuracy/test_acc': acc}

        result = pl.EvalResult()
        result.log_dict(logs, prog_bar=True)
        return result

    def test_epoch_end(self, outputs: List[Any]) -> None:

        fig = plot_distributions(self.trainer.datamodule.st_distribution)

        self.logger[0].experiment.add_figure('Spatial-Temporal Distribution',
                                             fig)

        avg_loss, avg_acc, mean_ap, cmc = self.evaluation_metrics(outputs)

        mAP_logs = {'Results/test_mAP': mean_ap,
                    'Results/test_CMC_top1': cmc[0].tolist(),
                    'Results/test_CMC_top5': cmc[4].tolist()}

        log = {'Results/test_loss': avg_loss.tolist(),
               'Results/test_accuracy': avg_acc.tolist(),
               }

        out = {**log, **mAP_logs}

        result = pl.EvalResult()
        result.log_dict(out)
        return result
