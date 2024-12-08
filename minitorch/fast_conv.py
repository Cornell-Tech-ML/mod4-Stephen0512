from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """A decorator that JIT compiles a function for parallel execution."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

        `batch, in_channels, width`

    and weight tensor

        `out_channels, in_channels, k_width`

    Computes padded output of

        `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `weight` tensor.
        weight_shape (Shape): shape for `weight` tensor.
        weight_strides (Strides): strides for `weight` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # Calculate the result for each output position in parallel
    for out_pos in prange(out_size):
        # Declare an new index array for the current output position
        out_index: Index = np.empty(MAX_DIMS, np.int32)

        # Convert the flat index of the current output position to the corresponding tensor indices
        to_index(out_pos, out_shape, out_index)

        # Extract the batch, output channel, and width indices from the output tensor indices
        out_batch, out_channel, out_width = out_index[0], out_index[1], out_index[2]

        # Calculate the flat index of the current output position for further usages
        out_pos_index = index_to_position(out_index, out_strides)

        # Declare a variable to calculate the sum of the dot product of the input and weight tensors for the current output position
        result = 0.0

        # Calculate dot product between each element in the input window and the weight kernel
        for in_channel in prange(in_channels):
            for width_index in prange(kw):
                # Determine weight offset and input width based on the direction of the weight (reverse or not)
                if reverse:
                    weight_width_offset = (
                        kw - 1 - width_index
                    )  # Reverse calculated the weight width offset
                    input_width = (
                        out_width - weight_width_offset
                    )  # Calculated the input width by moving the input width to the left
                else:
                    weight_width_offset = (
                        width_index  # Forward calculated the weight width offset
                    )
                    input_width = (
                        out_width + weight_width_offset
                    )  # Calculated the input width by moving the input width to the right

                # Check if the input index position is within the edges (Skip if not as 0 is used for out of bound input)
                if input_width >= 0 and input_width < width:
                    # Calculate the flat indices of the current input position
                    input_pos = (
                        out_batch * s1[0] + in_channel * s1[1] + input_width * s1[2]
                    )

                    # Calculate the flat indices of the current weight position
                    weight_pos = (
                        out_channel * s2[0]
                        + in_channel * s2[1]
                        + weight_width_offset * s2[2]
                    )

                    # Calculate the dot product of the input and weight and add it to the result
                    result += input[input_pos] * weight[weight_pos]

        # Store the result in the corresponding position in the output tensor
        out[out_pos_index] = result


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute a 1D Convolution Backward

        Args:
        ----
            ctx : Context
            grad_output : batch x out_channel x h x w

        Returns:
        -------
            Tuple[Tensor, Tensor] containing:
                grad_input : Gradient with respect to the input tensor, shape (batch x in_channel x h x w)
                grad_weight : Gradient with respect to the weight tensor, shape (out_channel x in_channel x kh x kw)

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

        `batch, in_channels, height, width`

    and weight tensor

        `out_channels, in_channels, k_height, k_width`

    Computes padded output of

        `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `weight` tensor.
        weight_shape (Shape): shape for `weight` tensor.
        weight_strides (Strides): strides for `weight` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Declare intermediate variable for strides of input, weight, and output tensors
    s1 = input_strides
    s2 = weight_strides

    # Extract the individual strides for each dimension of input, weight, and output tensors
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # Calculate the result for each output position in parallel
    for out_pos in prange(out_size):
        # Declare an new index array for the current output position
        out_index: Index = np.empty(MAX_DIMS, np.int32)

        # Convert the flat index of the current output position to the corresponding tensor indices
        to_index(out_pos, out_shape, out_index)

        # Extract the batch, output channel, and width indices from the output tensor indices
        out_batch, out_channel, out_height, out_width = (
            out_index[0],
            out_index[1],
            out_index[2],
            out_index[3],
        )

        # Calculate the flat index of the current output position for further usages
        out_pos_index = index_to_position(out_index, out_strides)

        # Declare a variable to calculate the sum of the dot product of the input and weight tensors for the current output position
        result = 0.0

        # Calculate dot product between each element in the input window and the weight kernel
        for in_channel in prange(in_channels):
            for height_index in prange(kh):
                for width_index in prange(kw):
                    # Determine weight offsets and input dimensions based on the direction of the weight (reverse or not)
                    if reverse:
                        weight_height_offset = (
                            kh - 1 - height_index
                        )  # Reverse calculated the weight height offset
                        weight_width_offset = (
                            kw - 1 - width_index
                        )  # Reverse calculated the weight width offset
                        input_height = (
                            out_height - weight_height_offset
                        )  # Calculated the input height by moving the input height to the left
                        input_width = (
                            out_width - weight_width_offset
                        )  # Calculated the input width by moving the input width to the left
                    else:
                        weight_height_offset = (
                            height_index  # Forward calculated the weight height offset
                        )
                        weight_width_offset = (
                            width_index  # Forward calculated the weight width offset
                        )
                        input_height = (
                            out_height + weight_height_offset
                        )  # Calculated the input height by moving the input height to the right
                        input_width = (
                            out_width + weight_width_offset
                        )  # Calculated the input width by moving the input width to the right

                    # Check if the input height and width index position are within the edges (Skip if not as 0 is used for out of bound input)
                    if (
                        input_height >= 0
                        and input_height < height
                        and input_width >= 0
                        and input_width < width
                    ):
                        # Calculate the flat indices of the current input position
                        input_pos = (
                            out_batch * s10
                            + in_channel * s11
                            + input_height * s12
                            + input_width * s13
                        )

                        # Calculate the flat indices of the current weight position
                        weight_pos = (
                            out_channel * s20
                            + in_channel * s21
                            + weight_height_offset * s22
                            + weight_width_offset * s23
                        )

                        # Calculate the dot product of the input and weight and add it to the result
                        result += input[input_pos] * weight[weight_pos]

        # Store the result in the corresponding position in the output tensor
        out[out_pos_index] = result


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute a 2D Convolution Backward

        Args:
        ----
            ctx : Context
            grad_output : batch x out_channel x h x w

        Returns:
        -------
            Tuple[Tensor, Tensor] containing:
                grad_input : Gradient with respect to the input tensor, shape (batch x in_channel x h x w)
                grad_weight : Gradient with respect to the weight tensor, shape (out_channel x in_channel x kh x kw)

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
