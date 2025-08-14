# sparse-nqs

Interpretable scaling behavior in sparse subnetwork representations of quantum states.

This repository contains the code used for the experiments in [Interpretable Scaling Behavior in Sparse Subnetwork Representations of Quantum States](https://arxiv.org/abs/2505.22734).

---

## Setup

We recommend using a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

We provide a minimal set of environment dependencies to reproduce the main results. Additional Python packages may be required to regenerate final, publication-ready figures.

---

## Reproducibility

All experiments are configured via scripts in the `example_scripts/` directory. These scripts are designed to:
- Demonstrate core training and pruning pipelines
- Run on standard laptops (≥16 GB RAM) for small systems
- Scale to larger systems with GPU acceleration

Note: All hyperparameters and experiment-specific settings are documented in the Appendix. You will need to update the arguments in the example scripts to fully reproduce our results.

### Step-by-step Instructions

1. Set file paths  
   Update file paths in the scripts within `example_scripts/` to point to your desired directories for model checkpoints and output data.

2. Run IMP-WR (Iterative Magnitude Pruning with Weight Rewinding)  
   We provide example IMP-WR scripts for the following combinations:
   - Transverse Field Ising Model (TFIM): FFNN, CNN, ResCNN
   - Toric Code: FFNN

3. Train sparse networks in isolation  
   Using the masks and weights from the pruning phase, run isolated training for:
   - T(theta_init, m_imp)
   - T(theta_rand, m_imp)
   - T(theta_init, m_rand)

   The `pruner.py` module automatically loads the correct initialization and mask for each setting via the variational Monte Carlo (VMC) driver.

4. Compute fidelity between networks  
   To analyze the overlap between neural quantum states at different sparsity levels, use the fidelity routines provided in the `figure_notebooks/` directory.

---

## Model Architectures

The following prunable network architectures are implemented:

1. Prunable Feedforward Neural Network (PrunableFFNN)  
   - Supported pruning protocols: `layerwise`, `random_layerwise`

2. Prunable Convolutional Neural Network (PrunableCNN)  
   - Supported pruning protocols: `layerwise`, `random_layerwise`

3. Prunable Residual CNN (PrunableResCNN)  
   - Supported pruning protocols: `res_blocks_global`, `res_blocks_random_global`

---

For additional implementation details, refer to the inline documentation in `pruner.py`, `vmc.py`, and the configuration scripts in `example_scripts/`. The pruning routines are provided in the `networks/` directory.
