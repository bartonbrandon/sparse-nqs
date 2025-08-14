# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import opt_einsum

from flax.core.frozen_dict import FrozenDict
from flax import nnx
from flax.nnx import rnglib, variablelib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
  # PromoteDtypeFn,
)

Array = jax.Array
Axis = int
Size = int


default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()
# default_mask_init = initializers.ones()


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """ "Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, tp.Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    ' int or pair of ints.'
  )


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class Linear(Module):
  """A linear transformation applied over the last dimension of the input.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> layer = nnx.Linear(in_features=3, out_features=4, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': VariableState(
        type=Param,
        value=(4,)
      ),
      'kernel': VariableState(
        type=Param,
        value=(3, 4)
      )
    })

  Args:
    in_features: the number of input features.
    out_features: the number of output features.
    use_mask: whether to apply a mask to the weights.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    dot_general: dot product function.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
      and a ``dtype`` keyword argument, and return a tuple of arrays with the
      promoted dtype.
    rngs: rng key.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_mask: bool = True,
    use_bias: bool = False,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float64,
    precision: PrecisionLike = None,
    kernel_init: Initializer = initializers.normal(0.1),
    mask_init: Initializer = initializers.ones,
    bias_init: Initializer = default_bias_init,
    dot_general: DotGeneralT = lax.dot_general,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    rngs: rnglib.Rngs,
  ):
    kernel_key = rngs.params()
    self.kernel = nnx.Param(
      kernel_init(kernel_key, (in_features, out_features), param_dtype)
    )

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_key = rngs.params()
      self.bias = nnx.Param(
        bias_init(bias_key, (out_features,), param_dtype)
        )
    else:
      self.bias = None

    self.mask: nnx.Variable[jax.Array] | None
    if use_mask:
        mask_key = rngs.params()
        self.mask = nnx.Variable(
          mask_init(mask_key, (in_features, out_features), param_dtype)
          )
    else:
        self.mask = None

    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.dot_general = dot_general
    self.promote_dtype = promote_dtype

  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.kernel.value
    mask = self.mask.value
    bias = self.bias.value if self.bias is not None else None

    if self.mask is not None:
      kernel = jnp.multiply(kernel, mask)

    inputs, kernel, bias = self.promote_dtype(
      (inputs, kernel, bias), dtype=self.dtype
    )

    y = self.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    assert self.use_bias == (bias is not None)
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    return y


class Conv(Module):
  """Convolution Module wrapping ``lax.conv_general_dilated``.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> rngs = nnx.Rngs(0)
    >>> x = jnp.ones((1, 8, 3))

    >>> # valid padding
    >>> layer = nnx.Conv(in_features=3, out_features=4, kernel_size=(3,),
    ...                  padding='VALID', rngs=rngs)
    >>> layer.kernel.value.shape
    (3, 3, 4)
    >>> layer.bias.value.shape
    (4,)
    >>> out = layer(x)
    >>> out.shape
    (1, 6, 4)

    >>> # circular padding with stride 2
    >>> layer = nnx.Conv(in_features=3, out_features=4, kernel_size=(3, 3),
    ...                  strides=2, padding='CIRCULAR', rngs=rngs)
    >>> layer.kernel.value.shape
    (3, 3, 3, 4)
    >>> layer.bias.value.shape
    (4,)
    >>> out = layer(x)
    >>> out.shape
    (1, 4, 4)

    >>> # apply lower triangle mask
    >>> mask = jnp.tril(jnp.ones((3, 3, 4)))
    >>> layer = nnx.Conv(in_features=3, out_features=4, kernel_size=(3,),
    ...                  mask=mask, padding='VALID', rngs=rngs)
    >>> out = layer(x)

  Args:
    in_features: int or tuple with number of input features.
    out_features: int or tuple with number of output features.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer, which will be interpreted
      as a tuple of the single integer. For all other cases, it must be a
      sequence of integers.
    strides: an integer or a sequence of ``n`` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), the string `'REFLECT'`
      (reflection across the padding boundary), or a sequence of ``n``
      ``(low, high)`` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides. ``'CAUSAL'`` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of ``inputs``
      (default: 1). Convolution with input dilation ``d`` is equivalent to
      transposed convolution with stride ``d``.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
      and a ``dtype`` keyword argument, and return a tuple of arrays with the
      promoted dtype.
    rngs: rng key.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: int | tp.Sequence[int],
    strides: tp.Union[None, int, tp.Sequence[int]] = 1,
    *,
    padding: PaddingLike = 'SAME',
    input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    use_mask: bool = True,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float64,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    mask_init: Initializer = initializers.ones,
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    rngs: rnglib.Rngs,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)
    else:
      kernel_size = tuple(kernel_size)

    kernel_shape = kernel_size + (
      in_features // feature_group_count,
      out_features,
    )
    kernel_key = rngs.params()
    self.kernel_shape = kernel_shape
    self.kernel = nnx.Param(kernel_init(kernel_key, kernel_shape, param_dtype))

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_shape = (out_features,)
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
    else:
      self.bias = None

    self.mask: nnx.Variable[jax.Array] | None
    if use_mask:
        mask_shape = kernel_shape
        mask_key = rngs.params()
        self.mask = nnx.Variable(mask_init(mask_key, mask_shape, param_dtype))
    else:
        self.mask = None

    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.feature_group_count = feature_group_count
    self.use_bias = use_bias
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.conv_general_dilated = conv_general_dilated
    self.promote_dtype = promote_dtype

  def __call__(self, inputs: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions ``(*batch_dims, spatial_dims..., features)``.
        This is the channels-last convention, i.e. NHWC for a 2d convolution and
        NDHWC for a 3D convolution. Note: this is different from the input convention
        used by ``lax.conv_general_dilated``, which puts the spatial dimensions last.
        Note: If the input has more than 1 batch dimension, all batch dimensions
        are flattened into a single dimension for the convolution and restored
        before returning.  In some cases directly vmap'ing the layer may yield
        better performance than this default flattening approach.  If the input
        lacks a batch dimension it will be added for the convolution and removed
        n return, an allowance made to enable writing single-example code.

    Returns:
      The convolved data.
    """

    assert isinstance(self.kernel_size, tuple)
    kernel_size = self.kernel_size

    def maybe_broadcast(
      x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
    ) -> tuple[int, ...]:
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    # Combine all input batch dimensions into a single leading batch axis.
    num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
        num_batch_dimensions:
      ]
      inputs = jnp.reshape(inputs, flat_input_shape)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(kernel_size))
    if padding_lax in ('CIRCULAR', 'REFLECT'):
      assert isinstance(padding_lax, str)
      kernel_size_dilated = [
        (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: tp.List[tuple[int, int]] = [(0, 0)]
      pads = (
        zero_pad
        + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
        + [(0, 0)]
      )
      padding_mode = {'CIRCULAR': 'wrap', 'REFLECT': 'reflect'}[padding_lax]
      inputs = jnp.pad(inputs, pads, mode=padding_mode)
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(kernel_size) != 1:
        raise ValueError(
          'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs = jnp.pad(inputs, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    # One shared convolutional kernel for all pixels in the output.
    assert self.in_features % self.feature_group_count == 0

    if self.mask is not None and self.mask.shape != self.kernel_shape:
      raise ValueError(
        'Mask needs to have the same shape as weights. '
        f'Shapes are: {self.mask.shape}, {self.kernel_shape}'
      )

    kernel = self.kernel.value
    mask = self.mask.value
    if mask is not None:
      kernel = jnp.multiply(kernel, mask)

    bias = self.bias.value if self.bias is not None else None

    inputs, kernel, bias = self.promote_dtype(
      (inputs, kernel, bias), dtype=self.dtype
    )

    y = self.conv_general_dilated(
      inputs,
      kernel,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    if self.use_bias:
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)  # type: ignore
      y += bias

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
      
    return y


