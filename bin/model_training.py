import logging
import os
import shutil

import gin

from bin.run_job_utils import ExperimentConfig
from sss.trainers import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@gin.configurable
def run_model_training(
    config: ExperimentConfig,
    trainer: BaseTrainer,
    seed: int = 0,
):
    os.makedirs(config.model_dir, exist_ok=True)

    # copy the config file to the base directory
    shutil.copy(config.job_config, config.base_dir)
    shutil.copy(config.model_config, config.base_dir)
    shutil.copy(config.trainer_config, config.base_dir)

    trainer = trainer(model_config=config.model_config, model_saving_path = config.model_dir)
    trainer.init()
    trainer.distributed_training()
