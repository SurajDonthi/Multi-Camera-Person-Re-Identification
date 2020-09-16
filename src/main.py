from argparse import ArgumentParser
import os
from pathlib2 import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from .engine import Engine
from .data import CustomDataLoader


def main():
    tb_logger = TensorBoardLogger(save_dir=args.log_path, name="")

    checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(checkpoint_dir,
                                     #  monitor='val_loss',
                                     #  save_last=True,
                                     #  mode='min',
                                     #  save_top_k=1
                                     )

    data_loader = CustomDataLoader(data_dir=args.data_path,
                                   #    batch_size=args.batch_size,
                                   #    test_batch_size=args.test_batch_size
                                   )

    model = Engine(None)

    trainer = Trainer.from_argparse_args(args, logger=tb_logger,
                                         checkpoint_callback=chkpt_callback,
                                         #   early_stop_callback=False,
                                         #   weights_summary='full',
                                         #   gpus=1,
                                         #   max_epochs=20
                                         )

    trainer.fit(model, data_loader)
    # trainer.test(model)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = Engine.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
