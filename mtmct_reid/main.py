import os
from argparse import ArgumentParser

from pathlib2 import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.test_tube import TestTubeLogger

from data import ReIDDataModule
from engine import ST_ReID
from utils import save_args


def main(args):
    tt_logger = TestTubeLogger(save_dir=args.log_path, name="",
                               description=args.description,
                               create_git_tag=True,
                               debug=args.debug
                               )
    tt_logger.experiment

    log_dir = Path(tt_logger.save_dir) / f"version_{tt_logger.version}"

    checkpoint_dir = log_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(checkpoint_dir,
                                     monitor='Loss/val_loss',
                                     save_last=True,
                                     mode='min',
                                     save_top_k=10,
                                     )

    data_module = ReIDDataModule.from_argparse_args(args)

    model = ST_ReID(data_module.num_classes, learning_rate=args.learning_rate,
                    criterion=args.criterion, rerank=args.rerank)

    save_args(args, log_dir)

    trainer = Trainer.from_argparse_args(args, logger=[tt_logger],
                                         checkpoint_callback=chkpt_callback,
                                         profiler=True)  # AdvancedProfiler()

    trainer.fit(model, data_module)
    trainer.test(model)     # , data_module


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = ST_ReID.add_model_specific_args(parser)
    parser = ReIDDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print(f'\nArguments: \n{args}\n')

    main(args)
