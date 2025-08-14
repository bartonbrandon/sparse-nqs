import json
import os
import flax
from flax import nnx
import equinox as eqx
import jax.numpy as jnp
import jax

from ..networks.prunable_ffnn import PrunableFFNN
from ..networks.prunable_res_cnn import PrunableResCNN
from ..networks.prunable_cnn import PrunableCNN

def save_dict_as_json(dict, save_data_path):
    """
    Save the hyperparameters to a json file.
    """
    file_name = os.path.expanduser(save_data_path + 'hyperparams.json')
    with open(file_name, "wb") as f:
        hyperparam_str = json.dumps(dict)
        f.write((hyperparam_str + "\n").encode())
    return

def load_dict_from_json(load_data_path):
    """
    Load the hyperparameters from a json file.
    """
    file_name = os.path.expanduser(load_data_path + 'hyperparams.json')
    with open(file_name, "rb") as f:
        hyperparam_str = f.read().decode()
        dict = json.loads(hyperparam_str)
    return dict


def save_training_log(training_log, save_data_path, pruning_iter):
    """
    Save the training log to a json file.
    """
    file_name = os.path.expanduser(save_data_path + 'training_log_piter={}.json'.format(pruning_iter))
    training_log.serialize(file_name)
    return

def save_sampling_log(sampling_log, save_data_path, pruning_iter):
    """
    Save the sampling log to a json file.
    """
    file_name = os.path.expanduser(save_data_path + 'sampling_log_piter={}.json'.format(pruning_iter))
    sampling_log.serialize(file_name)
    return


def serialize_network(network, save_data_path, pruning_iter):
    """ Serialize the network at a given pruning step
    """
    network_hyperparams = network.get_hyperams()

    # Separate parameters and state
    params, state = nnx.split(network)

    file_name = os.path.expanduser(save_data_path + r'model_piter={}.eqx'.format(pruning_iter))
    with open(file_name, "wb") as f:
        hyperparam_str = json.dumps(network_hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, state)


def load_serialized_network(init_path, pruning_iter, network_type):
    """ Load a network from a file
    """
    file_name = os.path.expanduser(init_path + r'model_piter={}.eqx'.format(pruning_iter))

    with open(file_name, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        
        if network_type == 'PrunableFFNN':
            network = PrunableFFNN(**hyperparams)
        elif network_type == 'PrunableResCNN':
            network = PrunableResCNN(**hyperparams)
        elif network_type == 'PrunableCNN':
            network = PrunableCNN(**hyperparams)
        else:
            raise ValueError('Network type not recognized')
        
        graph, old_state = nnx.split(network)
        new_state = eqx.tree_deserialise_leaves(f, old_state)
        new_model = nnx.merge(graph, new_state)
        
    return new_model
