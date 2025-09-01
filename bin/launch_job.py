import gin
from clize import run
import os

from bin.data_generation import run_data_generation
from bin.subdomain_data_cropping import run_subdomain_data_cropping
from bin.model_training import run_model_training
from bin.iterative_subdomain_solve import run_iterative_subdomain_solve
from bin.DDM_global_solve import run_DDM_global_solve
from bin.run_job_utils import ExperimentConfig, set_up_experiment
from bin.inverse_design import run_inverse_design

@gin.configurable
def run_job(
    job_config: str,
    pipeline: str,
    experiment_name: str,
    experiment_dir: str=None,
    trainer_config: str=None,
    model_config: str=None,
    design_config: str=None,
    data_config: str=None,
    iterative_config: str=None,
    gpu_ids: str="0,1"
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    unique_exp_name, base_dir, log_dir, code_dir, output_dir, model_dir = set_up_experiment(
        name=experiment_name,
        base_dir=experiment_dir,
    )

    config = ExperimentConfig(
        job_config=job_config,
        design_config=design_config,
        experiment_dir=base_dir,
        experiment_name=unique_exp_name,
        trainer_config=trainer_config,
        model_config=model_config,
        data_config=data_config,
        iterative_config=iterative_config,
        base_dir=base_dir,
        log_dir=log_dir,
        code_dir=code_dir,
        output_dir=output_dir,
        model_dir=model_dir,
    )
    if pipeline == "data_generation":
        job_fn = run_data_generation
    elif pipeline == "train":
        job_fn = run_model_training
    elif pipeline == "iterative_subdomain":
        job_fn = run_iterative_subdomain_solve
    elif pipeline == "DDM_global":
        job_fn = run_DDM_global_solve
    elif pipeline == "inverse_design":
        job_fn = run_inverse_design
    elif pipeline == "subdomain_data_cropping":
        job_fn = run_subdomain_data_cropping
    else:
        raise ValueError(f"Pipeline {pipeline} not found")

    job_fn(config)

def main(*,
    job_config: str,
    experiment_name: str,
    design_config: str=None,
    trainer_config: str=None,
    model_config: str=None,
    data_config: str=None,
    iterative_config: str=None,
    pipeline: str):

    gin.parse_config_file(job_config)

    if design_config is not None:
        gin.parse_config_file(design_config)
    if trainer_config is not None:
        gin.parse_config_file(trainer_config)
    if model_config is not None:
        gin.parse_config_file(model_config)
    if data_config is not None:
        gin.parse_config_file(data_config)
    if iterative_config is not None:
        gin.parse_config_file(iterative_config)

    run_job(
        job_config=job_config,
        pipeline=pipeline,
        experiment_name=experiment_name,
        design_config=design_config,
        trainer_config=trainer_config,
        model_config=model_config,
        data_config=data_config,
        iterative_config=iterative_config,
    )


if __name__ == "__main__":
    run(main)