import logging
import os
import shutil

import gin

from bin.run_job_utils import ExperimentConfig
from sss.iterative.subdomain_solver import SubdomainSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@gin.configurable
def run_iterative_subdomain_solve(
    config: ExperimentConfig,
    seed: int = 0,
):
    # copy the config file to the base directory
    shutil.copy(config.job_config, config.base_dir)
    shutil.copy(config.iterative_config, config.base_dir)

    solver = SubdomainSolver(
        output_dir=config.output_dir,
    )

    solver.init()
    solver.solve_from_dataset()
