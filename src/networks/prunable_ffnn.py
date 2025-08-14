import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import opt_einsum
from src.networks.network_utils import nnx_linear_layers

from flax.core.frozen_dict import FrozenDict
from flax import nnx
from flax.nnx import rnglib, variablelib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
    Any,
    Dtype,
    Shape,
    Initializer,
    PrecisionLike,
    DotGeneralT,
    ConvGeneralDilatedT,
    PaddingLike,
    LaxPadding,
  #   PromoteDtypeFn,
)

Array = jax.Array
Axis = int
Size = int

default_kernel_init = initializers.normal()
default_bias_init = initializers.zeros_init()

class PrunableFFNN(Module):
    """ A RBM ....

    Example usage::

    Args:
      
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rngs_key: int,
        *,
        use_bias: bool = False,
        use_kernel_mask: bool = True,
        activation: Any = nnx.relu,
        param_dtype: Dtype = jnp.float64,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
    ):
        rngs = nnx.Rngs(rngs_key)

        # Initialize the linear layer
        self.linear = nnx_linear_layers.Linear(in_features=input_dim, 
                                        out_features=hidden_dim,
                                        use_bias=use_bias,
                                        use_mask=use_kernel_mask,
                                        param_dtype=param_dtype,
                                        precision=precision,
                                        kernel_init=kernel_init,
                                        bias_init=bias_init,
                                        rngs=rngs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rngs_key = rngs_key
        self.use_bias = use_bias
        self.use_kernel_mask = use_kernel_mask
        self.activation = activation
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init


    def __call__(self, x: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """

        # Apply the linear layer
        y = self.linear(x)

        # Activation
        y = self.activation(y)

        # Sum output
        y = jnp.sum(y, axis=-1)

        return y
    

    """ Pruning utils """

    def prune(self, protocol, pr):
        """
        Prune the network according to the protocol.

        Args:
            protocol: str, pruning protocol
            pr: float, pruning ratio
        """
        # Regular pruning
        valid_protocols = ['layerwise', 'random_layerwise']
        if protocol in valid_protocols:
            self.prune_with_protocol(protocol, pr)
        else:
            raise ValueError("Invalid pruning protocol. Options: 'layerwise', 'random_layerwise'")

    
    def layerwise_prune_generic_kernel(self, kernel, mask, pr):
        """
        Generic kernel pruning method.
        
        Args:
            kernel: jnp.array, kernel weights
            mask: jnp.array, kernel mask
            pr: float, pruning ratio
        """

        # mask the weights in the kernel
        masked_kernel = jnp.multiply(kernel, mask)

        # Flatten the kernel to compute the a layer-wise threshold
        non_zero_weights = masked_kernel[masked_kernel != 0]
        non_zero_weights = non_zero_weights.flatten()

        # Find the index of the kth largest element
        k = jnp.rint(pr * non_zero_weights.size).astype(int)
        
        # Get the threshold by indexing the kth largest element
        if (k>0):
            threshold = jnp.sort(jnp.abs(non_zero_weights))[k-1]
        else:
            threshold = jnp.sort(jnp.abs(non_zero_weights))[0]
        
        # Update the mask using the threshold
        new_mask = jnp.where(jnp.abs(masked_kernel) > threshold, 1.0, 0.0)
        new_kernel = jnp.multiply(kernel, new_mask)

        return new_kernel, new_mask
    
    def random_prune_generic_kernel(self, kernel, mask, pr):
        """
        Generic kernel pruning method.

        Args:
            kernel: jnp.array, kernel weights
            mask: jnp.array, kernel mask
            pr: float, pruning ratio

        Returns:
            new_kernel: jnp.array, pruned kernel
            new_mask: jnp.array, pruned mask
        """
        # Identify non-zero weights (i.e., currently unpruned)
        flat_mask = mask.flatten()
        nonzero_indices = jnp.nonzero(flat_mask, size=flat_mask.size, fill_value=-1)[0]
        valid_indices = nonzero_indices[nonzero_indices != -1]

        n_prune = jnp.rint(pr * valid_indices.size).astype(int)

        # Initialize new mask
        new_mask = flat_mask

        if n_prune > 0:
            rngs = nnx.Rngs(self.rngs_key)
            key = rngs.param()
        
            prune_indices = jax.random.choice(key, valid_indices, shape=(n_prune,), replace=False)
            
            new_mask = new_mask.at[prune_indices].set(0.0)
        else:
            print("No weights to prune.")
        
        new_mask = new_mask.reshape(mask.shape)
        new_kernel = jnp.multiply(kernel, new_mask)

        return new_kernel, new_mask


    def prune_with_protocol(
            self,
            protocol:str,
            pr:float):
        """
        Prune the kernel of the network layerwise, in-place.
        
        Args:
            protocol: str, pruning protocol
            pr: float, pruning ratio
        """

        kernel = self.linear.kernel.value
        mask = self.linear.mask.value

        if protocol == 'layerwise':
            new_kernel, new_mask = self.layerwise_prune_generic_kernel(kernel, mask, pr)
        elif protocol == 'random_layerwise':
            new_kernel, new_mask = self.random_prune_generic_kernel(kernel, mask, pr)
        
        # Update the kernel and mask in place
        self.linear.kernel.value = new_kernel
        self.linear.mask.value = new_mask
        
        return

    
    def rewind_weights(self, rewind_model):
        """
        Rewind the weights of the current model with the weights of a different model.
        Alternatively, you can update the masks in the rewind model with the current model's masks using update_masks().

        Args:
            rewind_model: PrunableRBM, model to rewind the weights from
        """
        # Update the embedding layer weights
        self.linear.kernel.value = rewind_model.linear.kernel.value

        return
    
    def update_masks(self, update_model):
        """
        Update the masks of the current model with the masks of a different model.
        Alternatively, you can rewind the weights in the current model with the weights of the update model using rewind_weights().

        Args:
            update_model: PrunableRBM, model to update the masks from
        """
        # Update the masks
        self.linear.mask.value = update_model.linear.mask.value

        return
    

    def get_num_params(self):
        """
        Get the number of parameters in the model.
        """
        num_params = jnp.count_nonzero(self.linear.mask.value)

        return num_params


    def get_hyperams(self):
        """
        Get the hyperparameters of the model.
        """
        hyperparams = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'rngs_key': self.rngs_key
        }
        return hyperparams
    




