import logging
import os
import shutil

import gin

from bin.run_job_utils import ExperimentConfig
from sss.iterative.global_solver import GlobalSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@gin.configurable
def run_DDM_global_solve(
    config: ExperimentConfig,
    seed: int = 0,
):
    shutil.copy(config.job_config, config.base_dir)
    shutil.copy(config.iterative_config, config.base_dir)

    solver = GlobalSolver(
        output_dir=config.output_dir,
        seed=seed,
    )

    solver.init()
    solver.solve_from_random()
