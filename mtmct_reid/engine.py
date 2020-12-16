from argparse import ArgumentParser
from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import str_to_bool
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.sgd import SGD

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
                 criterion: str = 'cross_entropy', rerank: bool = False,
                 save_features: bool = True):

        super().__init__(num_classes)

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.criterion = LOSSES[criterion]
        self.rerank = rerank
        # self.save_hyperparameters()

    @staticmethod   # @classmethod not required as
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-lp', '--log_path', type=str, required=False,
                            default='./logs')
        parser.add_argument('-lr', '--learning_rate', required=False,
                            type=float, default=0.1)
        parser.add_argument('-c', '--criterion', type=str, required=False,
                            choices=LOSSES.keys(),
                            default='cross_entropy')
        parser.add_argument('-re', '--rerank', required=False,
                            type=bool, default=False)
        parser.add_argument('-sfe', '--save_features', required=False,
                            type=bool, default=False)
        parser.add_argument('-des', '--description', required=False, type=str)
        parser.add_argument('--git-tag', required=False, type=str_to_bool)
        parser.add_argument('--debug', required=False, type=str_to_bool)
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

    def prepare_data(self) -> None:
        self.st_distribution = self.trainer.datamodule.st_distribution

    # Shared steps -> To calculate loss/acc & uses parts_probs
    def shared_step(self, batch):
        X, y = batch
        parts_proba = self(X, training=True)    # returns a list of parts

        loss = 0
        y_pred = torch.zeros_like(parts_proba[0])
        for part in parts_proba:
            loss += self.criterion(part, y)
            y_pred += F.softmax(part, dim=1)

        y_pred = torch.stack(parts_proba, dim=0).sum(dim=0)
        _, y_pred = torch.max(y_pred, dim=1)
        acc = torch.mean((y == y_pred).type(torch.float16))

        return loss, acc

    # Training
    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        logs = {'Loss/train_loss': loss, 'Accuracy/train_acc': acc}
        self.log_dict(logs, prog_bar=True)
        return loss

    # Validation
    def on_validation_epoch_start(self) -> None:

        self.q_features = torch.Tensor()
        self.q_targets = torch.Tensor()
        self.q_cam_ids = torch.Tensor()
        self.q_frames = torch.Tensor()
        self.g_features = torch.Tensor()
        self.g_targets = torch.Tensor()
        self.g_cam_ids = torch.Tensor()
        self.g_frames = torch.Tensor()

    def eval_shared_step(self, batch, dataloader_idx):
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

    def validation_step(self, batch, batch_idx, dataloader_idx):

        if dataloader_idx == 2:
            loss, acc = self.shared_step(batch)

            self.val_logs = {'Loss/val_loss': loss, 'Accuracy/val_acc': acc}
            self.log_dict(self.val_logs, prog_bar=True)
            # return loss
        else:
            self.eval_shared_step(batch, dataloader_idx)
            if not hasattr(self, 'val_logs'):
                self.val_logs = {'Loss/val_loss': 0, 'Accuracy/val_acc': 0}
            self.log_dict(self.val_logs, logger=False)

    def evaluation_metrics(self):

        self.q_features = l2_norm_standardize(self.q_features)
        self.g_features = l2_norm_standardize(self.g_features)

        # Scores against all feature vectors in the gallery image for each
        # image in the query data.
        scores = joint_scores(self.q_features,
                              self.q_cam_ids,
                              self.q_frames,
                              self.g_features,
                              self.g_cam_ids,
                              self.g_frames,
                              self.trainer.datamodule.st_distribution)

        if self.rerank:
            scores = re_ranking(scores)

        # Metrics & Evaluation
        mean_ap, cmc = mAP(scores, self.q_targets,
                           self.q_cam_ids,
                           self.g_targets,
                           self.g_cam_ids)

        return mean_ap, cmc

    def validation_epoch_end(self, outputs):
        mean_ap, cmc = self.evaluation_metrics()

        mAP_logs = {'Results/val_mAP': mean_ap,
                    'Results/val_CMC_top1': cmc[0].tolist(),
                    'Results/val_CMC_top5': cmc[4].tolist()}
        self.log_dict(mAP_logs)

    # Testing
    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.eval_shared_step(batch, dataloader_idx)

    def test_epoch_end(self, outputs):

        fig = plot_distributions(self.trainer.datamodule.st_distribution)

        self.logger.experiment.add_figure('Spatial-Temporal Distribution',
                                          fig)

        mean_ap, cmc = self.evaluation_metrics()

        mAP_logs = {'Results/test_mAP': mean_ap,
                    'Results/test_CMC_top1': cmc[0].tolist(),
                    'Results/test_CMC_top5': cmc[4].tolist()}

        self.log_dict(mAP_logs)
