import torch
from utils.setting import Setting
from utils.trainer import Trainer
import sys

if __name__ == '__main__':

    args, logger = Setting().run()

    trainer = Trainer(args, logger)

    for epoch in range(args.epochs):
        logger.info(f'Start Training Epoch {epoch}')
        trainer.train_epoch(epoch)
        logger.info(f'Finish Training Epoch {epoch}')

    logger.info('Training Finished')
