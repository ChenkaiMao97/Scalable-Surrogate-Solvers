# Scalable Surrogate Solvers (sss)

## Environment setup

Install micromamba, which is a drop-in replacement for conda, but it resolves dependencies using a C library (and thus is orders of magnitude faster than conda). (Do this in your home or root directory)

In your <home_directory> or <root_directory>, run the following commands:
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba shell init -s bash -r micromamba
```

Your ~/.bashrc should be updated, so you can update your PATH by running:

```bash
source ~/.bashrc
```

Create a new environment with the dependencies.

```bash
micromamba env create -f environment.yml
```

Activate the environment.

```bash
micromamba activate sss
```

## Run different pipelines:

### Full Domain Data generation
```bash
python -m bin.launch_job --job-config="bin/configs/cpu_job_config.gin" --experiment-name="ceviche_data_gen" --pipeline="data_generation" --data-config="bin/configs/data_gen/ceviche_multi_wl_dL.gin"
```

### Subdomain Data cropping
```bash
python -m bin.launch_job --job-config="bin/configs/cpu_job_config.gin" --experiment-name="crop_subdomain" --pipeline="subdomain_data_cropping" --data-config="bin/configs/data_gen/subdomain_cropping.gin"
```

### Subdomain Model training
```bash
python -m bin.launch_job --job-config="bin/configs/multi_GPU_job_config.gin" --experiment-name="train_subdomain" --pipeline="train" --trainer-config="bin/configs/trainer/fixed_point_iteration_trainer.gin" --model-config="bin/configs/models/SM_FNO.gin"
```

MultiGrid inspired broadband model with and efficient setup / solve steps
```bash
python -m bin.launch_job --job-config="bin/configs/multi_GPU_job_config.gin" --experiment-name="train_subdomain_broadband_with_PML" --pipeline="train" --trainer-config="bin/configs/trainer/fixed_point_iteration_trainer_MG_broadband.gin" --model-config="bin/configs/models/MGUFO2d_broadband.gin"
```

with damping loss (simplified PML)
```bash
python -m bin.launch_job --job-config="bin/configs/multi_GPU_job_config.gin" --experiment-name="train_subdomain_broadband_damping" --pipeline="train" --trainer-config="bin/configs/trainer/fixed_point_iteration_trainer_MG_broadband_damping.gin" --model-config="bin/configs/models/MGUFO2d_broadband.gin"
```

training with GMRES:
```bash
python -m bin.launch_job --job-config="bin/configs/multi_GPU_job_config.gin" --experiment-name="train_subdomain_scaled_wl_dL" --pipeline="train" --trainer-config="bin/configs/trainer/trainer_GMRES.gin" --model-config="bin/configs/models/MGUFO2d_scaled_freq.gin"
```

### Subdomain solving with NN preconditioned iterative methods
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="iterative_subdomain" --pipeline="iterative_subdomain" --iterative-config="bin/configs/iterative/GMRES_MG.gin"
```

### Full domain Simulation with Domain Decomposition & coarse space
Schwarz inner region (without PML)
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="DDM_solve_inner" --pipeline="DDM_global" --iterative-config="bin/configs/iterative/two_level_overlapping_schwarz_MG_256_inner_domain.gin"
```

Schwarz full simulation
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="DDM_solve_full" --pipeline="DDM_global" --iterative-config="bin/configs/iterative/two_level_overlapping_schwarz_MG_256.gin"
```

Nonuniform Schwarz full simulation with direct solved PML
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="non_uniform_DDM_with_direct_PML_solve" --pipeline="DDM_global" --iterative-config="bin/configs/iterative/nonuniform_roll_schwarz_MG_256.gin"
```

Two level schwarz with coarse space (inner region no PML)
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="non_uniform_DDM_with_direct_PML_solve256" --pipeline="DDM_global" --iterative-config="bin/configs/iterative/two_level_coarse_space_overlapping_schwarz_MG_256.gin"
```

### Inverse Design

#### Wavelength division multiplexer (WDM)
with ceviche backend:
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="inverse_design_ceviche" --pipeline="inverse_design" --design-config="bin/configs/invde/wl_splitter_ceviche_TO.gin"
```

with DDM backend, 256 by 256 model:
```bash
python -m bin.launch_job --job-config="bin/configs/multi_GPU_job_config.gin" --experiment-name="inverse_design_DDM256" --pipeline="inverse_design" --design-config="bin/configs/invde/wl_splitter_DDM_TO.gin" --iterative-config="bin/configs/iterative/nonuniform_roll_schwarz_MG_256.gin"
```

#### Grating coupler
with ceviche backend:
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="inverse_design_ceviche_gc" --pipeline="inverse_design" --design-config="bin/configs/invde/grating_coupler_ceviche_TO.gin"
```

with DDM backend, 256 by 256 model:
```bash
python -m bin.launch_job --job-config="bin/configs/multi_GPU_job_config.gin" --experiment-name="inverse_design_DDM256_gc" --pipeline="inverse_design" --design-config="bin/configs/invde/grating_coupler_DDM_TO.gin" --iterative-config="bin/configs/iterative/nonuniform_roll_schwarz_MG_256.gin"
```


#### Metasurface
with ceviche backend:
```bash
python -m bin.launch_job --job-config="bin/configs/single_GPU_job_config.gin" --experiment-name="inverse_design_ceviche_meta" --pipeline="inverse_design" --design-config="bin/configs/invde/metasurface_ceviche_TO.gin"
```

with DDM backend, 256 by 256 model:
```bash
python -m bin.launch_job --job-config="bin/configs/multi_GPU_job_config.gin" --experiment-name="inverse_design_DDM256_meta" --pipeline="inverse_design" --design-config="bin/configs/invde/metasurface_DDM_TO.gin" --iterative-config="bin/configs/iterative/nonuniform_roll_schwarz_MG_256.gin"
```

## License

This project is licensed under the **Business Source License 1.1 (BUSL-1.1)**.  
Copyright © 2025 Chenkai Mao <chenkaim@stanford.edu>

- **Research and non-commercial use**: Permitted under the terms of BUSL-1.1.  
- **Commercial use**: Not permitted without a separate commercial license agreement.  
- **Change Date**: On the specified Change Date in the [LICENSE](./LICENSE) file, this code will automatically convert to an open source license (as defined in that file).  

For full details, please see the [LICENSE](./LICENSE) file in this repository.