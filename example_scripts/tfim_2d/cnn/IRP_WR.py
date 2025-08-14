import sys
sys.path.append('../../../')
from src import *
from src.hamiltonian_ops.tfim_2d import *
from src.vmc.vmc import *
from src.networks.prunable_cnn import *
from src.pruning_algorithm.pruner import *
from src.utils import hyperparam_utils

import netket as nk
import flax
from flax import nnx
import jax


Lx,Ly = 4,4
kappa = 3.04438
periodic_bc = False
# Sampling parameters
num_samples = 1024
num_chains = 16
chunk_size = None
num_discard = None
# Optimization parameters
learning_rate = 0.001
diag_shift = 0.001
# Network parameters
rngs_key = 2
kernel_size = 3
num_kernels = 4
# Pruning parameters
dense_epochs = 10000
iterative_epochs = 1000
pruning_protocol = 'random_layerwise'
pruning_ratio = 0.05
weight_rewinding = True
pruning_its = 40
sampling_its = 100
network_type = 'PrunableCNN'


ham_params = {
    'Lx': Lx,
    'Ly': Ly,
    'kappa': kappa,
    'periodic_bc': periodic_bc
    }

network_params = {
    'Lx': Lx,
    'kernel_size': kernel_size,
    'num_kernels': num_kernels,
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

save_data_path = '' # TODO: Insert your path to save data here

hyperparam_utils.save_dict_as_json(total_hyperparams, save_data_path)

# Load hyperparameters
params = hyperparam_utils.load_dict_from_json(save_data_path)

# Unpack hyperparameters
ham_params = params['ham_params']
network_params = params['network_params']
vmc_params = params['vmc_params']
pruner_params = params['pruner_params']


# Construct the Hamiltonian
operator = TFIM2D(**ham_params)

# Construct the neural network
network = PrunableCNN(**network_params)

# Construct the VMC wavefunction
vmc = VMC(operator, 
          network, 
          **vmc_params)

# Initialize the pruning algorithm
pruner = Pruner(vmc, **pruner_params)

init_path = '' # TODO: Insert path to IMP-WR data

# Iterative random pruning with weight rewinding starting from a dense pretrained network
pruner.IRP_WR(init_path=init_path, save_data=True, save_data_path=save_data_path)

