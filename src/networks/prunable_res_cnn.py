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

class ResBlock(Module):
    """Residual block with two convolutional layers."""
    
    def __init__(
        self, 
        num_kernels: int, 
        kernel_size: int,
        *,
        use_bias: bool = False,
        use_kernel_mask: bool = True, 
        activation: tp.Any = nnx.gelu,
        param_dtype: Dtype = jnp.float32, 
        kernel_init: Initializer = default_kernel_init,
        rngs: rnglib.Rngs,
    ):
        self.use_kernel_mask = use_kernel_mask
        self.layer_norm = nnx_layer_norm.LayerNorm(num_features=num_kernels, rngs=rngs)
        self.activation = activation
        self.use_kernel_mask = use_kernel_mask

        self.conv1 = nnx_linear_layers.Conv(
            in_features=num_kernels, 
            out_features=num_kernels, 
            kernel_size=(kernel_size,kernel_size),
            use_mask = use_kernel_mask,
            use_bias=use_bias,
            param_dtype=param_dtype, 
            kernel_init=kernel_init, 
            padding='SAME',
            rngs = rngs
        )
        
        self.conv2 = nnx_linear_layers.Conv(
            in_features=num_kernels, 
            out_features=num_kernels, 
            kernel_size=(kernel_size,kernel_size),
            use_mask = use_kernel_mask,
            use_bias=use_bias,
            param_dtype=param_dtype, 
            kernel_init=kernel_init, 
            padding='SAME',
            rngs = rngs
        )
        

    def __call__(self, x):
        residual = x

        x = self.layer_norm(x)
        
        x = self.activation(x)

        x = self.conv1(x)

        x = self.activation(x)

        x = self.conv2(x)
        
        return x + residual
    

class PrunableResCNN(Module):
    """Prunable Residual CNN."""

    def __init__(
        self,
        *,
        Lx: int,
        num_res_blocks: int, 
        kernel_size: int, 
        num_kernels: int,
        use_bias: bool = False,
        use_kernel_mask: bool = True,
        activation: tp.Any = nnx.gelu,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        dot_general: DotGeneralT = lax.dot_general,
        rngs_key: int = 0,
    ):
        self.num_res_blocks = num_res_blocks
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.Lx = Lx
        self.use_kernel_mask = use_kernel_mask
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

        # Stack of ResBlocks
        self.res_blocks = [
            ResBlock(
                num_kernels=num_kernels, kernel_size=kernel_size, 
                use_kernel_mask=use_kernel_mask, activation=activation,
                  param_dtype=param_dtype, kernel_init=kernel_init, rngs=rngs
            ) for _ in range(num_res_blocks)
        ]

        self.final_layer_norm = nnx_layer_norm.LayerNorm(num_features=num_kernels, rngs=rngs)

    def __call__(self, x):
        
        if len(x.shape) == 2:
            x = x.reshape((-1, self.Lx, self.Lx, 1)) # (batch, height, width, channels=1) [channels = 3 for RGB values]

        # Embed the input
        x = self.embedding_layer(x)
        
        # Apply the ResBlocks
        for block in self.res_blocks:
            x = block(x)

        # LayerNorm
        x = self.final_layer_norm(x)
  
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
        if protocol == 'embedding_layerwise':
            self.prune_embedding_layerwise(pr)
        elif protocol == 'res_blocks_layerwise':
            self.prune_res_blocks_layerwise(pr)
        elif protocol == 'blockwise':
            self.prune_res_blocks_blockwise(pr)
        elif protocol == 'res_blocks_global':
            self.prune_res_blocks_global(pr)
        elif protocol == 'hybrid':
            self.prune_hybrid(pr)
        # Random pruning
        elif protocol == 'embedding_random_layerwise':
            self.prune_embedding_random_layerwise(pr)
        elif protocol == 'random_res_blocks_layerwise':
            self.prune_res_blocks_random_layerwise(pr)
        elif protocol == 'random_blockwise':
            self.prune_res_blocks_random_blockwise(pr)
        elif protocol == 'res_blocks_random_global':
            self.prune_res_blocks_random_global(pr)
        elif protocol == 'random_hybrid':
            self.prune_random_hybrid(pr)
        else:
            raise ValueError("Invalid pruning protocol.")


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

        new_mask.astype(dtype=jnp.float32)
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
    

    def global_prune_generic_kernels(self, kernels, masks, pr):
        """
        Generic global kernel pruning method for a list of kernels.

        Args:
            kernels: List[jnp.array], kernel weights
            masks: List[jnp.array], kernel mask
            pr: float, pruning ratio
        """

        # Mask the weights in the kernel
        masked_kernels = jnp.array([jnp.multiply(kernel, mask) for kernel, mask in zip(kernels, masks)])  

        # Flatten the kernel to compute a global threshold
        non_zero_weights = masked_kernels[masked_kernels != 0].flatten()
        
        if non_zero_weights.size == 0:
            return ValueError("All masks are zero. Cannot prune.")

        # Find the pruning threshold
        k = max(1, min(non_zero_weights.size, jnp.rint(pr * non_zero_weights.size).astype(int)))
        threshold = jnp.sort(jnp.abs(non_zero_weights))[k - 1]

        # Vectorized mask update
        new_masks = [jnp.where(jnp.abs(k) > threshold, 1.0, 0.0) for k in masked_kernels]
        new_kernels = [jnp.multiply(k, m) for k, m in zip(kernels, new_masks)]

        return new_kernels, new_masks
    

    def random_prune_generic_kernels(self, kernels, masks, pr):
        """
        Generic global kernel pruning method for a list of kernels.

        Args:
            kernels: List[jnp.array], kernel weights
            masks: List[jnp.array], kernel mask
            pr: float, pruning ratio
        """
        # Identify non-zero weights (i.e., currently unpruned)
        masks = jnp.array(masks)
        flat_masks = masks.flatten()
        nonzero_indices = jnp.nonzero(flat_masks, size=flat_masks.size, fill_value=-1)[0]
        valid_indices = nonzero_indices[nonzero_indices != -1]

        n_prune = jnp.ceil(pr * valid_indices.size).astype(int)

        # Initialize new mask
        new_mask = flat_masks

        if n_prune > 0:
            rngs = nnx.Rngs(self.rngs_key)
            key = rngs.param()
            prune_indices = jax.random.choice(key, valid_indices, shape=(n_prune,), replace=False)
            new_mask = new_mask.at[prune_indices].set(0.0)
        else:
            print("No weights to prune.")
        
        new_masks = new_mask.reshape(masks.shape)
        new_kernels = [jnp.multiply(k, m) for k, m in zip(kernels, new_masks)]

        return new_kernels, new_masks
    

    def prune_embedding_layerwise(self, pr):
        """
        Prune the embedding layer of the network, in-place.
        
        Args:
            pr: float, pruning ratio
        """

        kernel = self.embedding_layer.kernel.value
        mask = self.embedding_layer.mask.value

        new_kernel, new_mask = self.layerwise_prune_generic_kernel(kernel, mask, pr)
        
        # Update the kernel and mask in place
        self.embedding_layer.kernel.value = new_kernel
        self.embedding_layer.mask.value = new_mask.astype(dtype=jnp.float32) 
        return
    
    
    def prune_embedding_random_layerwise(self, pr):
        """
        Prune the embedding layer of the network, in-place.
        
        Args:
            pr: float, pruning ratio
        """

        kernel = self.embedding_layer.kernel.value
        mask = self.embedding_layer.mask.value

        new_kernel, new_mask = self.random_prune_generic_kernel(kernel, mask, pr)
        
        # Update the kernel and mask in place
        self.embedding_layer.kernel.value = new_kernel
        self.embedding_layer.mask.value = new_mask.astype(dtype=jnp.float32) 
        return
    

    def prune_res_block_layerwise(self, block_index, pr):
        """
        Prune a single res block of the network, in-place.
        This method prunes the convolutional layers of the res block layerwise.
        
        Args:
            layer_index: int, index of the res block to prune
            pr: float, pruning ratio
        """
        # Access the first convolutional layer
        kernel = self.res_blocks[block_index].conv1.kernel.value
        mask = self.res_blocks[block_index].conv1.mask.value
        # Prune the first convolutional layer
        new_kernel, new_mask = self.layerwise_prune_generic_kernel(kernel, mask, pr)
        # Update the kernel and mask in place
        self.res_blocks[block_index].conv1.kernel.value = new_kernel
        self.res_blocks[block_index].conv1.mask.value = new_mask.astype(dtype=jnp.float32) 

        # Access the second convolutional layer
        kernel = self.res_blocks[block_index].conv2.kernel.value
        mask = self.res_blocks[block_index].conv2.mask.value
        # Prune the second convolutional layer
        new_kernel, new_mask = self.layerwise_prune_generic_kernel(kernel, mask, pr)
        # Update the kernel and mask in place
        self.res_blocks[block_index].conv2.kernel.value = new_kernel
        self.res_blocks[block_index].conv2.mask.value = new_mask.astype(dtype=jnp.float32) 
        return
    
    
    def prune_res_block_random_layerwise(self, block_index, pr):
        """
        Prune a single res block of the network, in-place.
        This method prunes the convolutional layers of the res block layerwise.
        
        Args:
            layer_index: int, index of the res block to prune
            pr: float, pruning ratio
        """
        # Access the first convolutional layer
        kernel = self.res_blocks[block_index].conv1.kernel.value
        mask = self.res_blocks[block_index].conv1.mask.value
        # Prune the first convolutional layer
        new_kernel, new_mask = self.random_prune_generic_kernel(kernel, mask, pr)
        # Update the kernel and mask in place
        self.res_blocks[block_index].conv1.kernel.value = new_kernel
        self.res_blocks[block_index].conv1.mask.value = new_mask.astype(dtype=jnp.float32) 

        # Access the second convolutional layer
        kernel = self.res_blocks[block_index].conv2.kernel.value
        mask = self.res_blocks[block_index].conv2.mask.value 
        # Prune the second convolutional layer
        new_kernel, new_mask = self.random_prune_generic_kernel(kernel, mask, pr)
        # Update the kernel and mask in place
        self.res_blocks[block_index].conv2.kernel.value = new_kernel
        self.res_blocks[block_index].conv2.mask.value = new_mask.astype(dtype=jnp.float32) 
        return
    

    def prune_blockwise(self, block_index, pr):
        """
        Prune a single res block of the network, in-place.
        This method prunes the convolutional layers of the res block globally.
        
        Args:
            block_index: int, index of the res block to prune
            pr: float, pruning ratio
        """
        # Access the first convolutional layer
        conv1_kernel = self.res_blocks[block_index].conv1.kernel.value
        conv1_mask = self.res_blocks[block_index].conv1.mask.value.astype(dtype=jnp.float32) 
        # Access the second convolutional layer
        conv2_kernel = self.res_blocks[block_index].conv2.kernel.value
        conv2_mask = self.res_blocks[block_index].conv2.mask.value.astype(dtype=jnp.float32) 

        kernels = [conv1_kernel, conv2_kernel]
        masks = [conv1_mask, conv2_mask]
        
        # Prune blockwise
        new_kernels, new_masks = self.global_prune_generic_kernels(kernels, masks, pr)

        # Update the kernel and mask in place
        self.res_blocks[block_index].conv1.kernel.value = new_kernels[0]
        self.res_blocks[block_index].conv1.mask.value = new_masks[0].astype(dtype=jnp.float32)
        self.res_blocks[block_index].conv2.kernel.value = new_kernels[1]
        self.res_blocks[block_index].conv2.mask.value = new_masks[1].astype(dtype=jnp.float32)        
        return
    

    def prune_random_blockwise(self, block_index, pr):
        """
        Prune a single res block of the network, in-place.
        This method prunes the convolutional layers of the res block globally.
        
        Args:
            block_index: int, index of the res block to prune
            pr: float, pruning ratio
        """
        # Access the first convolutional layer
        conv1_kernel = self.res_blocks[block_index].conv1.kernel.value
        conv1_mask = self.res_blocks[block_index].conv1.mask.value
        # Access the second convolutional layer
        conv2_kernel = self.res_blocks[block_index].conv2.kernel.value
        conv2_mask = self.res_blocks[block_index].conv2.mask.value

        kernels = [conv1_kernel, conv2_kernel]
        masks = [conv1_mask, conv2_mask]
        
        # Prune blockwise
        new_kernels, new_masks = self.random_prune_generic_kernels(kernels, masks, pr)

        # Update the kernel and mask in place
        self.res_blocks[block_index].conv1.kernel.value = new_kernels[0]
        self.res_blocks[block_index].conv1.mask.value = new_masks[0].astype(dtype=jnp.float32)
        self.res_blocks[block_index].conv2.kernel.value = new_kernels[1]
        self.res_blocks[block_index].conv2.mask.value = new_masks[1].astype(dtype=jnp.float32)        
        return


    def prune_res_blocks_layerwise(self, pr):
        """
        Prune the res blocks of the network layerwise.
        
        Args:
            pr: float, pruning ratio
        """
        for i in range(len(self.res_blocks)):
            self.prune_res_block_layerwise(i, pr)

        return
    
    def prune_res_blocks_random_layerwise(self, pr):
        """
        Prune the res blocks of the network layerwise.
        
        Args:
            pr: float, pruning ratio
        """
        for i in range(len(self.res_blocks)):
            self.prune_res_block_random_layerwise(i, pr)

        return
    
    def prune_res_blocks_blockwise(self, pr):
        """
        Prune the res blocks of the network blockwise.
        
        Args:
            pr: float, pruning ratio
        """
        for i in range(len(self.res_blocks)):
            self.prune_blockwise(i, pr)
        return
    
    def prune_res_blocks_random_blockwise(self, pr):
        """
        Prune the res blocks of the network blockwise.
        
        Args:
            pr: float, pruning ratio
        """
        for i in range(len(self.res_blocks)):
            self.prune_random_blockwise(i, pr)
        return


    def prune_res_blocks_global(self, pr):
        """
        Prune the res blocks of the network globally.

        Args:
            pr: float, pruning ratio
        """
        conv_kernels = []
        conv_masks = []

        for i in range(len(self.res_blocks)):
            conv_kernels.append(self.res_blocks[i].conv1.kernel.value)
            conv_masks.append(self.res_blocks[i].conv1.mask.value)
            conv_kernels.append(self.res_blocks[i].conv2.kernel.value)
            conv_masks.append(self.res_blocks[i].conv2.mask.value)

        kernels, masks = self.global_prune_generic_kernels(conv_kernels, conv_masks, pr)

        for i in range(0, len(self.res_blocks)):
            self.res_blocks[i].conv1.kernel.value = kernels[2*i]
            self.res_blocks[i].conv1.mask.value = masks[2*i].astype(dtype=jnp.float32)
            self.res_blocks[i].conv2.kernel.value = kernels[(2*i)+1]
            self.res_blocks[i].conv2.mask.value = masks[(2*i)+1].astype(dtype=jnp.float32)
        return
    

    def prune_res_blocks_random_global(self, pr):
        """
        Prune the res blocks of the network globally.

        Args:
            pr: float, pruning ratio
        """
        conv_kernels = []
        conv_masks = []

        for i in range(len(self.res_blocks)):
            conv_kernels.append(self.res_blocks[i].conv1.kernel.value)
            conv_masks.append(self.res_blocks[i].conv1.mask.value)
            conv_kernels.append(self.res_blocks[i].conv2.kernel.value)
            conv_masks.append(self.res_blocks[i].conv2.mask.value)

        kernels, masks = self.random_prune_generic_kernels(conv_kernels, conv_masks, pr)

        for i in range(0, len(self.res_blocks)):
            self.res_blocks[i].conv1.kernel.value = kernels[2*i]
            self.res_blocks[i].conv1.mask.value = masks[2*i].astype(dtype=jnp.float32)
            self.res_blocks[i].conv2.kernel.value = kernels[(2*i)+1]
            self.res_blocks[i].conv2.mask.value = masks[(2*i)+1].astype(dtype=jnp.float32)
        return
    
    
    def prune_hybrid(self, pr):
        """
            Prunes the Res Blocks globally, and the embedding layer layerwise.
        """
        
        # Embedding layer pruning
        self.prune_embedding_layerwise(pr*0.1)
        # Res Block global pruning
        self.prune_res_blocks_global(pr)

        return
    
    
    def prune_random_hybrid(self, pr):
        """
            Prunes the Res Blocks globally, and the embedding layer layerwise.
        """
        # Embedding layer pruning
        self.prune_embedding_random_layerwise(pr*0.1)
        # Res Block global pruning
        self.prune_res_blocks_random_global(pr)

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

        # Iterate over the res blocks
        for i in range(len(self.res_blocks)):
            # Update the weights of the convolutional layers
            self.res_blocks[i].conv1.kernel.value = rewind_model.res_blocks[i].conv1.kernel.value
            self.res_blocks[i].conv2.kernel.value = rewind_model.res_blocks[i].conv2.kernel.value
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

        # Iterate over the res blocks
        for i in range(len(self.res_blocks)):
            # Update the masks of the convolutional layers
            self.res_blocks[i].conv1.mask.value = current_model.res_blocks[i].conv1.mask.value
            self.res_blocks[i].conv2.mask.value = current_model.res_blocks[i].conv2.mask.value
        return


    def mask_entire_res_block(self, block_index):
        """
        Mask the entire res block.
        
        Args:
            block_index: int, index of the res block to mask
        """
        self.res_blocks[block_index].conv1.mask.value = jnp.zeros_like(self.res_blocks[block_index].conv1.mask.value)
        self.res_blocks[block_index].conv2.mask.value = jnp.zeros_like(self.res_blocks[block_index].conv2.mask.value)
        return


    def mask_all_res_blocks(self):
        """
        Mask all res blocks in the network.
        """
        for i in range(len(self.res_blocks)):
            self.mask_entire_res_block(i)
        return
    

    def mask_entire_embedding_layer(self):
        """
        Mask the entire embedding layer.
        """
        self.embedding_layer.mask.value = jnp.zeros_like(self.embedding_layer.mask.value)
        return
    

    def mask_all_layers(self):
        """
        Mask all layers in the network.
        """
        self.mask_all_res_blocks()
        self.mask_entire_embedding_layer()
        return
    

    def get_num_params(self):
        """
        Get the number of parameters in the model.
        """
        embedding_params = self.get_num_params_in_embedding_layer()
        res_block_params = self.get_num_params_in_res_blocks()

        return (embedding_params, res_block_params)
    

    def get_num_params_in_embedding_layer(self):
        """
        Get the number of parameters in the embedding layer.
        """
        num_params = jnp.count_nonzero(self.embedding_layer.mask.value)
        return num_params
    
    
    def get_num_params_in_res_blocks(self):
        """
        Get the number of parameters in the res blocks.
        """
        num_params = 0
        for block in self.res_blocks:
            num_params += jnp.count_nonzero(block.conv1.mask.value)
            num_params += jnp.count_nonzero(block.conv2.mask.value)

        return num_params
    
    def get_hyperams(self):
        """
        Get the hyperparameters of the model.
        """
        hyperparams = {
            'Lx': self.Lx,
            'num_res_blocks': self.num_res_blocks,
            'kernel_size': self.kernel_size,
            'num_kernels': self.num_kernels,
            'rngs_key': self.rngs_key
        }
        return hyperparams
