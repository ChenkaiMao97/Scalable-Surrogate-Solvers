import logging
import os
import shutil

import gin
import jax
import numpy as onp

from bin.run_job_utils import ExperimentConfig, seeding

from sss.invde.opt import CevicheDesign
from sss.invde.utils.utils import get_ceviche_general_block_challenge, get_ceviche_grating_coupler_challenge, get_ceviche_metasurface_challenge

design_schemes = ["general_block", 'grating_coupler', 'metasurface']

@gin.configurable
def run_inverse_design(
        config: ExperimentConfig,
        design_challenge,
        seed_ids=0
    ):
    shutil.copy(config.job_config, config.base_dir)
    shutil.copy(config.design_config, config.base_dir)
    if config.iterative_config is not None:
        shutil.copy(config.iterative_config, config.base_dir)

    if design_challenge not in design_schemes:
        raise ValueError(f"Design challenge {design_challenge} not found")

    key = jax.random.PRNGKey(seeding([seed_ids]))

    design_scheme_kwargs = {}
    if design_challenge == 'general_block':
        design_scheme_kwargs = {
                "challenge": get_ceviche_general_block_challenge(key=key)
            }
    elif design_challenge == 'grating_coupler':
        design_scheme_kwargs = {
                "challenge": get_ceviche_grating_coupler_challenge(key=key)
            }
    elif design_challenge == 'metasurface':
        design_scheme_kwargs = {
                "challenge": get_ceviche_metasurface_challenge(key=key)
            }
    design = CevicheDesign(log_dir=config.log_dir, **design_scheme_kwargs)
    # build the design
    design.init(key=key)
    results = design.optimize()
    design.stop_workers()

    input_eps, Ez_out_forward_RI, source_RI, wls, dLs, state = results

    os.makedirs(config.output_dir, exist_ok=True)
    onp.save(os.path.join(config.output_dir, "input_eps.npy"), input_eps)
    onp.save(os.path.join(config.output_dir, "Ez_out_forward_RI.npy"), Ez_out_forward_RI)
    onp.save(os.path.join(config.output_dir, "source_RI.npy"), source_RI)
    onp.save(os.path.join(config.output_dir, "wls.npy"), wls)
    onp.save(os.path.join(config.output_dir, "dLs.npy"), dLs)
    onp.save(os.path.join(config.output_dir, "loss_curve.npy"), state.loss)

if __name__ == "__main__":
    run(run_design)
