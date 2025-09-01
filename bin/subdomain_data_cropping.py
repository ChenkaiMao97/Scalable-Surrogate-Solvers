import logging
import os
import shutil

import gin

from bin.run_job_utils import ExperimentConfig
from sss.data.subdomain_cropper import SubdomainCropper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@gin.configurable
def run_subdomain_data_cropping(
    config: ExperimentConfig,
    seed: int = 0,
):
    print(config.base_dir, config.data_config)
    os.makedirs(config.base_dir, exist_ok=True)
    shutil.copy(config.job_config, config.base_dir)
    shutil.copy(config.data_config, config.base_dir)

    data_cropper = SubdomainCropper(output_dir=config.output_dir, seed=seed)
    data_cropper.crop_all()