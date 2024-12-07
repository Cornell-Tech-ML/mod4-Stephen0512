from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Calculate new dimensions by dividing the original dimensions by the kernel size
    new_h_dim = height // kh
    new_w_dim = width // kw

    # Ensure the input tensor is in contiguous memory layout for efficient reshaping
    input = input.contiguous()

    # Reshape the tensor to split height and width into blocks of size kh x kw
    # New shape: (batch, channel, height/kh, kh, width/kw, kw)
    reshaped_input = input.view(batch, channel, new_h_dim, kh, new_w_dim, kw)

    # Reorder dimensions to get the desired shape
    # From: (batch, channel, height/kh, kh, width/kw, kw)
    # To: (batch, channel, height/kh, width/kw, kh, kw)
    reordered_input = reshaped_input.permute(0, 1, 2, 4, 3, 5)

    # Ensure the reordered input tensor is in contiguous memory layout for efficient reshaping
    reordered_input = reordered_input.contiguous()

    # Reshape the tensor to the final shape by flattening the last two dimensions (kh, kw)
    # New shape: (batch, channel, height/kh, width/kw, kh*kw)
    result = reordered_input.view(batch, channel, new_h_dim, new_w_dim, kh * kw)

    # Return the result tensor and the new dimensions
    return result, new_h_dim, new_w_dim


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies 2D average pooling over an input tensor.

    Args:
    ----
        input: A input tensor with shape batch x channel x height x width
        kernel: A tuple of (kernel_height, kernel_width) specifying the pooling window size

    Returns:
    -------
        A result tensor with shape batch x channel x height/kernel_height x width/kernel_width, containing the average values of each kernel window in the last dimension.

    """
    # Get the shape of the input tensor for future usages
    batch, channel, _, _ = input.shape

    # Reshape the input tensor into tiles using the tile function implemented above
    reshaped_input, new_h_dim, new_w_dim = tile(input, kernel)

    # Calculate the mean over the last dimension with the size of (kernel_height * kernel_width)
    pooled_input = reshaped_input.mean(dim=4)

    # Ensure the pooled input tensor is in contiguous memory layout for efficient reshaping
    pooled_input = pooled_input.contiguous()

    # Reshape the pooled input tensor to the expected output shape by removing the last dimension
    # New shape: (batch, channel, new_height, new_width)
    result = pooled_input.view(batch, channel, new_h_dim, new_w_dim)

    # Return the result tensor after pooling
    return result
