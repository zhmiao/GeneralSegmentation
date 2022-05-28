import os
import yaml
import numpy as np
import fire
from munch import Munch

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, CometLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from src.data.utils import get_dataset
from src.algorithms.utils import get_algorithm


def main(config='./configs/voc_plain_051622.yaml',
         project='VOC',
         gpus=-1, 
         logger_type='csv',
         session=0,
         np_threads=8,
         evaluate=None,
         seed=0):

    ############
    # Set gpus #
    ############
    gpus = gpus if torch.cuda.is_available() else None

    #############################
    # Set environment variables #
    #############################
    # Set numpy threads
    os.environ["OMP_NUM_THREADS"] = str(np_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(np_threads)
    os.environ["MKL_NUM_THREADS"] = str(np_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(np_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(np_threads)
    
    #######################
    # Load configurations #
    #######################
    with open(config) as f:
        conf = Munch(yaml.load(f, Loader=yaml.FullLoader))
    if gpus == -1:
        conf.batch_size = int(conf.batch_size / torch.cuda.device_count())

    pl.seed_everything(seed)

    ###########################
    # Load data and algorithm #
    ###########################
    dataset = get_dataset(conf.dataset_name, conf=conf)
    learner = get_algorithm(conf.algorithm, conf=conf)

    ###############
    # Load logger #
    ###############
    if logger_type == 'csv':
        logger = CSVLogger(
            save_dir='./log/{}'.format(conf.algorithm),
            prefix=project,
            name='{}_{}'.format(conf.algorithm, conf.conf_id),
            version=session
        )
    elif logger_type == 'comet':
        logger = CometLogger(
            api_key=os.environ.get('COMET_API_KEY'),
            save_dir='./log/{}'.format(conf.algorithm),
            project_name=project,  # Optional
            experiment_name='{}_{}_{}'.format(conf.algorithm, conf.conf_id, session),
        )

    ##################
    # Load callbacks #
    ##################
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_mean_IoU', mode='max', dirpath='./weights/{}'.format(conf.algorithm), save_top_k=1,
        filename='{}-{}'.format(conf.conf_id, session) + '-{epoch:02d}-{valid_mean_IoU:.2f}', verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    #################
    # Setup trainer #
    #################
    trainer = pl.Trainer(
        max_steps=conf.num_iters,
        check_val_every_n_epoch=1, 
        log_every_n_steps = conf.log_interval, 
        gpus=gpus,
        logger=None if evaluate is not None else logger,
        callbacks=[lr_monitor, checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=0,
        profiler='simple',
        enable_progress_bar=True,
    )

    #######
    # RUN #
    #######
    if evaluate is not None:
        trainer.validate(learner, datamodule=dataset, ckpt_path=evaluate)
    else:
        trainer.fit(learner, datamodule=dataset)

if __name__ == '__main__':
    fire.Fire(main)