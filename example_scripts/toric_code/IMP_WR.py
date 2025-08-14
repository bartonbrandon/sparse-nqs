import sys
sys.path.append('../../')
from src import *
from src.hamiltonian_ops.toric_code import *
from src.vmc.tc_vmc import *
from src.networks.prunable_ffnn import *
from src.pruning_algorithm.pruner import *
from src.utils import hyperparam_utils

import netket as nk
import flax
from flax import nnx

# System parameters
Lx,Ly = 3,3
periodic_bc = True
# Sampling parameters
num_samples = 1024
chunk_size = 1024
num_discard = 10
num_chains = 16
use_tc_sampler = True
# Optimization parameters
learning_rate = 0.008
diag_shift = 0.001
# Network parameters
input_dim = 2*Lx*Ly
hidden_dim = 16*input_dim
rngs_key = 2
# Pruning parameters
dense_epochs = 2000
iterative_epochs = 200
pruning_protocol = 'layerwise'
pruning_ratio = 0.12
weight_rewinding = True
pruning_its = 20
sampling_its = 10
network_type = 'PrunableFFNN'


ham_params = {
    'Lx': Lx,
    'Ly': Ly,
    'periodic_bc': periodic_bc
    }

network_params = {
    'input_dim': input_dim,
    'hidden_dim': hidden_dim,
    'rngs_key': rngs_key
    }

vmc_params = {
    'num_samples': num_samples,
    'chunk_size': chunk_size,
    'num_discard': num_discard,
    'num_chains': num_chains,
    'learning_rate': learning_rate,
    'diag_shift': diag_shift,
    }

pruner_params = {
    'dense_epochs': dense_epochs,
    'iterative_epochs': iterative_epochs,
    'pruning_protocol': pruning_protocol,
    'pruning_ratio': pruning_ratio,
    'weight_rewinding': weight_rewinding,
    'pruning_its': pruning_its,
    'sampling_its': sampling_its,
    'network_type': network_type,
    }

total_hyperparams = {
    'ham_params': ham_params,
    'network_params': network_params,
    'vmc_params': vmc_params,
    'pruner_params': pruner_params
    }

save_data_path = '' # TODO: Add a path to save data

hyperparam_utils.save_dict_as_json(total_hyperparams, save_data_path)

# Load hyperparameters
params = hyperparam_utils.load_dict_from_json(save_data_path)

# Unpack hyperparameters
ham_params = params['ham_params']
network_params = params['network_params']
vmc_params = params['vmc_params']
pruner_params = params['pruner_params']


# Construct the Hamiltonian
operator = ToricCode(**ham_params)

# Construct the neural network
network = PrunableFFNN(**network_params)

# Construct the VMC wavefunction
vmc = VMC(operator, 
          network,
          **vmc_params)

# Initialize the pruning algorithm
pruner = Pruner(vmc, **pruner_params)

# Train dense network
pruner.IMP_WR(save_data=True, save_data_path=save_data_path)