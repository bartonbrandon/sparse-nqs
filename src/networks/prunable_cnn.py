import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import opt_einsum
from src.networks.network_utils import nnx_linear_layers
from src.networks.network_utils import nnx_layer_norm

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)

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

default_kernel_init = initializers.he_normal()
default_bias_init = initializers.he_normal()


class PrunableCNN(Module):
    """Prunable CNN."""

    def __init__(
        self,
        *,
        Lx: int,
        kernel_size: int, 
        num_kernels: int,
        use_bias: bool = False,
        use_kernel_mask: bool = True,
        use_layer_norm: bool = False,
        activation: tp.Any = nnx.gelu,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        dot_general: DotGeneralT = lax.dot_general,
        rngs_key: int = 0,
    ):
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.Lx = Lx
        self.use_kernel_mask = use_kernel_mask
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.dot_general = dot_general
        self.rngs_key = rngs_key
        rngs = nnx.Rngs(rngs_key)

        # Embedding layer
        self.embedding_layer = nnx_linear_layers.Conv(
            in_features=1, 
            out_features=num_kernels, 
            kernel_size=(self.kernel_size, self.kernel_size),
            use_mask = self.use_kernel_mask,
            use_bias=use_bias,
            param_dtype=param_dtype, 
            kernel_init=kernel_init, 
            padding='SAME',
            rngs = rngs
        )

        if self.use_layer_norm:
            self.layer_norm = nnx_layer_norm.LayerNorm(num_features=num_kernels, rngs=rngs)

    def __call__(self, x):
        
        if len(x.shape) == 2:
            x = x.reshape((-1, self.Lx, self.Lx, 1)) # (batch, height, width, channels=1) [channels = 3 for RGB values]

        # Embed the input
        x = self.embedding_layer(x)

        # LayerNorm
        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.activation(x)
  
        # Sum the outputs
        x = jnp.sum(x, axis=[1,2,3])

        return x


    def debug_nan_check(self, x, where):
        def true_fn(x):
            jax.debug.print("⚠️ Found NaNs in tensor! place: {where}", where=where)
            return x
        def false_fn(x):
            return x
        
        return jax.lax.cond(jnp.isnan(x).any(), true_fn, false_fn, x)


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
            self.prune_embedding_layer(protocol, pr)
        else:
            raise ValueError(f"Invalid pruning protocol. Options: {valid_protocols}.")
     
        return


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
        new_mask.astype(dtype=jnp.float32)
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

        n_prune = jnp.ceil(pr * valid_indices.size).astype(int)

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
        new_mask.astype(dtype=jnp.float32)
        new_kernel = jnp.multiply(kernel, new_mask)

        return new_kernel, new_mask
    

    def prune_embedding_layer(
            self,
            protocol:str,
            pr:float):
        """
        Prune the embedding layer of the network, in-place.
        
        Args:
            protocol: str, pruning protocol
            pr: float, pruning ratio
        """

        kernel = self.embedding_layer.kernel.value
        mask = self.embedding_layer.mask.value

        if protocol == 'layerwise':
            new_kernel, new_mask = self.layerwise_prune_generic_kernel(kernel, mask, pr)

            # Update the kernel and mask in place
            self.embedding_layer.kernel.value = new_kernel
            self.embedding_layer.mask.value = new_mask.astype(jnp.float32)

        elif protocol == 'random_layerwise':
            new_kernel, new_mask = self.random_prune_generic_kernel(kernel, mask, pr)
            
            self.embedding_layer.kernel.value = new_kernel
            self.embedding_layer.mask.value = new_mask.astype(jnp.float32)

        else:
            print(f"Invalid pruning protocol: {protocol}.")

        return
    
    
    def rewind_weights(self, rewind_model):
        """
        Rewind the weights of the current model with the weights of a different model.
        Alternatively, you can update the masks in the rewind model with the current model's masks using update_masks().

        Args:
            rewind_model: PrunableResCNN, model to rewind the weights from
        """
        # Update the embedding layer weights
        self.embedding_layer.kernel.value = rewind_model.embedding_layer.kernel.value

        return
        

    def update_masks(self, current_model):
        """
        Update the rewind model with masks from a different model
        Alternatively, you can rewind the *weights* in the current model with rewind_weights().
        
        Args:
            current_model: PrunableResCNN, current model
        """
        # Update the embedding layer mask
        self.embedding_layer.mask.value = current_model.embedding_layer.mask.value

        return


    def mask_entire_embedding_layer(self):
        """
        Mask the entire embedding layer.
        """
        self.embedding_layer.mask.value = jnp.zeros_like(self.embedding_layer.mask.value)
        
        return
    

    def get_num_params(self):
        """
        Get the number of parameters in the model.
        """
        embedding_params = self.get_num_params_in_embedding_layer()

        return embedding_params
    

    def get_num_params_in_embedding_layer(self):
        """
        Get the number of parameters in the embedding layer.
        """
        num_params = jnp.count_nonzero(self.embedding_layer.mask.value)

        return num_params
    
    
    def get_hyperams(self):
        """
        Get the hyperparameters of the model.
        """
        hyperparams = {
            'Lx': self.Lx,
            'kernel_size': self.kernel_size,
            'num_kernels': self.num_kernels,
            'rngs_key': self.rngs_key
        }
        
        return hyperparams
