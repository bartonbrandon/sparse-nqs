import os
import sys
# os.environ["JAX_PLATFORM_NAME"] = "gpu"

import numpy as np
import flax as flax
import netket as nk
import time
import pandas as pd
import jax
from flax import nnx
import equinox as eqx
import json
from src.utils import hyperparam_utils

class Pruner:
    """
    Class for pruning a neural network quantum state.
    """
    
    def __init__(
            self,
            vmc,
            dense_epochs: int,
            iterative_epochs: int,
            pruning_protocol: str,
            weight_rewinding: bool,
            pruning_its: int,
            pruning_ratio: float,
            sampling_its: int,
            network_type: str,
            ):
        """
        Initialize the Pruner class with the given parameters.

        Args:
        -----
            vmc: VMC object
                The VMC object to be used for training and sampling
            dense_epochs: int
                Number of epochs to train the dense network
            iterative_epochs: int
                Number of epochs to train the sparse network
            pruning_protocol: str
                The pruning protocol to be used
            weight_rewinding: bool
                Whether to rewind the weights after pruning
            pruning_its: int
                Number of pruning iterations
            pruning_ratio: float
                The ratio of parameters to prune
            sampling_its: int
                Number of iterations to sample observables
            network_type: str
                The type of network to be used (e.g. 'PrunableFFNN', 'PrunableResCNN'.)
            
        Returns:
        --------
            None
        """
        self.vmc = vmc
        self.dense_epochs = dense_epochs
        self.iterative_epochs = iterative_epochs
        self.pruning_protocol = pruning_protocol
        self.weight_rewinding = weight_rewinding
        self.pruning_its = pruning_its
        self.pruning_ratio = pruning_ratio
        self.sampling_its = sampling_its
        self.network_type = network_type
        # Other parameters
        self.init_model = None
        self.rewind_model = None
        self.vmc_driver = vmc.set_up()
    
    
    def prune(self):
        # Prune the network
        current_network = self.vmc_driver.state.model   # Get the network from the VMC driver
        current_network.prune(protocol=self.pruning_protocol, pr=self.pruning_ratio)
            
        if (self.weight_rewinding): # Rewind weights
            if (self.rewind_model is None):
                raise ValueError("Rewind model not set. Please set the rewind model before rewinding weights.")
            
            current_network.rewind_weights(self.rewind_model) # Rewind the weights in the current model

        # Update the network in the VMC driver
        new_variables, _ = self.vmc_driver.state._model_framework.wrap(current_network)
        self.vmc_driver.state.variables = new_variables

        return


    def train(
            self,
            training_epochs,
            sampling_its,
            p_iter,
            save_data=False,
            save_data_path=None,
            ):
        """
        Train the network in one pruning iteration.
        """
        # Make logers for each pruning step
        training_log = nk.logging.RuntimeLog()
        
        # Training
        self.vmc_driver.run(n_iter=training_epochs, out=training_log, show_progress=True)
        
        if (save_data):
            hyperparam_utils.serialize_network(self.vmc_driver.state.model, save_data_path, pruning_iter=p_iter)
            hyperparam_utils.save_training_log(training_log, save_data_path, pruning_iter=p_iter)

        # Sample observables
        self.sample_observables(sampling_its, save_data=save_data, save_data_path=save_data_path, p_iter=p_iter)

        return
    

    def sample_observables(
            self,
            sampling_its,
            save_data=False,
            save_data_path=None,
            p_iter=None,
            ):
        
        """
        Sample observables after training.
        """
        sampling_log = nk.logging.RuntimeLog()
        self.vmc_driver.optimizer.learning_rate = 0    # Turn off learning for sampling
        self.vmc_driver.run(n_iter=sampling_its, out=sampling_log, show_progress=True, obs=self.vmc.operator.observable_dict)
        self.vmc_driver.optimizer.learning_rate = self.vmc.learning_rate    # Turn on learning again

        if (save_data):
            hyperparam_utils.save_sampling_log(sampling_log, save_data_path, pruning_iter=p_iter)

        return


    def train_dense(
            self, 
            save_data=False, 
            save_data_path=None,
            p_iter='dense'
        ):
        """
        Trains the network without pruning.
        Samples observables after convergence.

        Args:
        -----
            save_data: bool
                Whether to save the network parameters to a equinox file
            save_data_path: str
                Path to save the data to

        Returns:
        --------
            None
        """
        self.train(training_epochs=self.dense_epochs,
                sampling_its=self.sampling_its,
                p_iter=p_iter,
                save_data=save_data,
                save_data_path=save_data_path)

        return
    

    def prune_iteratively(
            self, 
            save_data=False, 
            save_data_path=None):
        """
        Iteratively prunes the network and samples observables after each pruning step.

        Args:
        -----
            save_data: bool
                Whether to save the network parameters to a equinox file
            save_data_path: str
                Path to save the data to
        
        Returns:
        --------
            None
        """

        for p_iter in range(self.pruning_its):
            
            st = time.time()
            # Prune the network
            self.prune()
            # Train the network
            self.train(training_epochs=self.iterative_epochs, 
                            p_iter=p_iter,
                            sampling_its=self.sampling_its, 
                            save_data=save_data, 
                            save_data_path=save_data_path)
            et = time.time()
            print(f'   Pruning iter {p_iter} completed in: {et-st} seconds')
            print('Params remaining', self.vmc_driver.state.model.get_num_params())
        
        return
    

    """ Iterative pruning methods """

    def IMP_WR(
            self,
            save_data:bool=False,
            save_data_path:str=None,
            init_path:str=None,
            ):
        """
        Iterative magnitude pruning with weight rewinding.
        """

        if (init_path is not None):
            init = hyperparam_utils.load_serialized_network(init_path=init_path, 
                                                            pruning_iter='init',
                                                            network_type=self.network_type)
            # Update the network in the VMC driver
            new_variables, _ = self.vmc_driver.state._model_framework.wrap(init)
            self.vmc_driver.state.variables = new_variables

        if (save_data):
            print('   Saving the initialization')
            hyperparam_utils.serialize_network(self.vmc_driver.state.model, save_data_path, pruning_iter='init')

        # Train the dense network
        print('   Training the dense network')
        self.train_dense(save_data=save_data, save_data_path=save_data_path)

        # Set the rewind model
        self.rewind_model = self.vmc_driver.state.model

        # Iteratively prune and train
        print('   Iterative pruning and training')
        self.prune_iteratively(save_data=save_data, save_data_path=save_data_path)

        return
    

    def IRP_WR(
            self,
            save_data:bool=False,
            save_data_path:str=None,
            init_path:str=None,
            ):
        """
        Iterative random pruning with weight rewinding, starting from a dense (pre-trained) network.
        """

        if (init_path is not None):
            init = hyperparam_utils.load_serialized_network(init_path=init_path, 
                                                            pruning_iter='dense',
                                                            network_type=self.network_type)
            # Update the network in the VMC driver
            new_variables, _ = self.vmc_driver.state._model_framework.wrap(init)
            self.vmc_driver.state.variables = new_variables

        if (save_data):
            print('   Saving the pre-trained initialization')
            hyperparam_utils.serialize_network(self.vmc_driver.state.model, save_data_path, pruning_iter='init')

        # Set the rewind model
        self.rewind_model = self.vmc_driver.state.model

        # Iteratively prune and train
        print('   Iterative pruning and training')
        self.prune_iteratively(save_data=save_data, save_data_path=save_data_path)

        return
    

    def IMP_CT(
            self,
            save_data:bool=False,
            save_data_path:str=None,
            init_path:str=None,
            ):
        """
        Iterative magntitude pruning with continued training (no WR) starting from a dense (pre-trained) network.
        """

        if (init_path is not None):
            init = hyperparam_utils.load_serialized_network(init_path=init_path, 
                                                            pruning_iter='dense',
                                                            network_type=self.network_type)
            # Update the network in the VMC driver
            new_variables, _ = self.vmc_driver.state._model_framework.wrap(init)
            self.vmc_driver.state.variables = new_variables

        if (save_data):
            print('   Saving the pre-trained initialization')
            hyperparam_utils.serialize_network(self.vmc_driver.state.model, save_data_path, pruning_iter='init')

        # Set the rewind model
        self.rewind_model = self.vmc_driver.state.model

        # Iteratively prune and train
        print('   Iterative pruning and training')
        self.prune_iteratively(save_data=save_data, save_data_path=save_data_path)

        return

    """ Isolated training """

    def T_init_imp(
            self,
            t_rewind_iteration:int,
            save_data_path:str,
            init_path:str,
            save_data:bool=True,
            ):
        """
        Sparse initialization from the same initialization as IMP-WR.
        Masks are used from the IMP-WR data at a selected pruning interation. 
        """
        # Load the t_rewind initialization
        init = hyperparam_utils.load_serialized_network(init_path=init_path, 
                                                        pruning_iter='init', 
                                                        network_type=self.network_type)

        # Load the T_rewind model from the selected pruning iteration
        t_rewind_model = hyperparam_utils.load_serialized_network(init_path=init_path, 
                                                                  pruning_iter=t_rewind_iteration,
                                                                  network_type=self.network_type)

        # Apply the masks from the T_rewind model to the current model
        init.update_masks(t_rewind_model)

            # Update the network in the VMC driver
        new_variables, _ = self.vmc_driver.state._model_framework.wrap(init)
        self.vmc_driver.state.variables = new_variables
        
        print('   Training the sparse initialized network')
        # Train the sparse model for equivalent number of epochs as the T_rewind model
        self.train_dense(save_data=save_data, save_data_path=save_data_path, p_iter=t_rewind_iteration)

        return


    def T_rand_imp(
            self,
            t_rewind_iteration:int,
            save_data_path:str,
            init_path:str,
            path_to_mask:str,
            save_data:bool=True,
            ):
        """
        Sparse initialization from a random initialization.
        Masks are used from the IMP-WR data at a selected pruning interation. 
        """
        # Load the a random initialization
        random_init = hyperparam_utils.load_serialized_network(init_path=init_path, 
                                                               pruning_iter='init', 
                                                               network_type=self.network_type)

        # Load the T_rewind model from the selected pruning iteration
        t_rewind_model = hyperparam_utils.load_serialized_network(init_path=path_to_mask, 
                                                                  pruning_iter=t_rewind_iteration,
                                                                  network_type=self.network_type)

        # Apply the masks from the T_rewind model to the current model
        random_init.update_masks(t_rewind_model)

            # Update the network in the VMC driver
        new_variables, _ = self.vmc_driver.state._model_framework.wrap(random_init)
        self.vmc_driver.state.variables = new_variables
        
        print('   Training the sparse initialized network')
        # Train the sparse model for equivalent number of epochs as the T_rewind model
        self.train_dense(save_data=save_data, save_data_path=save_data_path, p_iter=t_rewind_iteration)

        return


    def T_init_rand(
            self,
            pruning_protocol:str,
            starting_density:float,
            save_data_path:str,
            init_path:str,
            save_data:bool=True,
            ):
        """
        Train a sparse network with a random mask from the same initialization as IMP-WR.
        """
        # Load the t_rewind initialization
        init = hyperparam_utils.load_serialized_network(init_path=init_path, 
                                                        pruning_iter='init', 
                                                        network_type=self.network_type)
        pr = 1-starting_density
        init.prune(protocol=pruning_protocol, pr=pr)

            # Update the network in the VMC driver
        new_variables, _ = self.vmc_driver.state._model_framework.wrap(init)
        self.vmc_driver.state.variables = new_variables
        
        # Train the sparse model for equivalent number of epochs as the T_rewind model
        self.train_dense(save_data=save_data, save_data_path=save_data_path, p_iter=starting_density)
        
        return
