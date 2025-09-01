import os
import hashlib

from dataclasses import dataclass, field
from getpass import getuser
from datetime import datetime
from typing import Optional


DATETIME_TO_FILEPATH_FORMAT = "%m_%d_%yT%H_%M_%S"

@dataclass
class ExperimentConfig:
    job_config: str
    trainer_config: str = field(default_factory=str)
    model_config: str = field(default_factory=str)
    design_config: str = field(default_factory=str)
    data_config: str = field(default_factory=str)
    DDM_config: str = field(default_factory=str)
    iterative_config: str = field(default_factory=str)
    experiment_dir: str = field(default_factory=str)
    experiment_name: str = field(default_factory=str)
    base_dir: str = field(default_factory=str)
    log_dir: str = field(default_factory=str)
    code_dir: str = field(default_factory=str)
    output_dir: str = field(default_factory=str)
    model_dir: str = field(default_factory=str)

def set_up_experiment(
    name,
    base_dir: Optional[str] = None,
    checkpoint_dir = '/media/tmp0'
):
    ts = datetime.now().strftime(DATETIME_TO_FILEPATH_FORMAT)

    experiment_name = f"{name}-{ts}"
    base_dir = base_dir or f"{checkpoint_dir}/{getuser()}/checkpoints/sss/{experiment_name}"

    # create the base directory
    os.makedirs(base_dir, exist_ok=True)

    log_dir = f"{base_dir}/logs"
    output_dir = f"{base_dir}/generated_devices"
    code_dir = f"{base_dir}/code"
    model_dir = f"{base_dir}/models"

    print(f"Experiment Name: {experiment_name}")
    # print(f"Log path: {log_dir} Snapshot path: {code_dir} ")
    return experiment_name, base_dir, log_dir, code_dir, output_dir, model_dir

def seeding(ids):
    seeding_str = "_".join([str(id) for id in ids])
    hash_obj = hashlib.sha256(seeding_str.encode("utf-8"))
    seed = int(hash_obj.hexdigest(), 16) % (2**32)
    return seed