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


fast_max = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return a boolean tensor indicating where the maximum values occur along dimension.
    
    Args:
    ----
        input: Input tensor to find argmax over
        dim: Dimension to reduce along
        
    Returns:
    -------
        Boolean tensor with True at positions of maximum values along dimension

    """
    # Get maximum values along dimension using Max function
    max_vals = fast_max(input, dim)
    
    # Return boolean mask of where input equals the max values
    return max_vals == input


class Max(Function):
    """Max function: computes the max of elements along a given dimension"""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the max of elements along a given dimension.

        Args:
        ----
            ctx: The context object used to save values for the backward method.
            a: The input tensor.
            dim: The dimension along which to max. If None, max over all dimensions.

        Returns:
        -------
            The max of elements along the specified dimension.

        """
        # Get the dimension value to save for backward
        dim_val = int(dim.item())

        # Save the input tensor and dimension value for backward
        ctx.save_for_backward(a, dim_val)

        # Return the max of elements along the specified dimension
        return fast_max(a, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the local derivative with respect to the input.

        Args:
        ----
            ctx: The context object with saved values from the forward method.
            grad_output: The local derivative of the higher-order function.

        Returns:
        -------
            A tuple containing the local derivative with respect to the input and a placeholder float.

        """
        # Get the saved input tensor and dimension value
        (input_tensor, dim_val) = ctx.saved_values

        # Create mask of where maximum values occurred
        mask = argmax(input_tensor, dim_val)
        
        # Return the mask times the expanded gradient output
        return mask * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along the specified dimension.
    
    Args:
    ----
        input: Input tensor
        dim: Dimension to reduce over
        
    Returns:
    -------
        Tensor with maximum values along dimension

    """
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax along the specified dimension.
    
    The softmax function is defined as:
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)) for all j).
    
    This normalizes the input values to a probability distribution that sums to 1.

    Args:
    ----
        input: Input tensor
        dim: Dimension to compute softmax over
        
    Returns:
    -------
        Tensor with softmax applied along the specified dimension.

    """
    # Get max values along dimension for numerical stability
    max_vals = max(input, dim)
    
    # Subtract max values and compute exponentials
    shifted = input - max_vals
    exp_vals = shifted.exp()
    
    # Normalize by sum of exponentials
    sum_exp = exp_vals.sum(dim)
    return exp_vals / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along the specified dimension.
    
    Args:
    ----
        input: Input tensor
        dim: Dimension to compute log softmax over
        
    Returns:
    -------
        Tensor with logsoftmax applied along the specified dimension.

    """
    # Compute softmax using the softmax function implemented above and then take the log
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies 2D max pooling over an input tensor.

    Args:
    ----
        input: A input tensor with shape batch x channel x height x width
        kernel: A tuple of (kernel_height, kernel_width) specifying the pooling window size

    Returns:
    -------
        A result tensor with shape batch x channel x height/kernel_height x width/kernel_width, containing the max values of each kernel window in the last dimension.

    """
    # Get the shape of the input tensor for future usages
    batch, channel, _, _ = input.shape

    # Reshape the input tensor into tiles using the tile function implemented above
    reshaped_input, new_h_dim, new_w_dim = tile(input, kernel)

    # Calculate the max over the last dimension with the size of (kernel_height * kernel_width)
    pooled_input = max(reshaped_input, 4)

    # Ensure the pooled input tensor is in contiguous memory layout for efficient reshaping
    pooled_input = pooled_input.contiguous()

    # Reshape the pooled input tensor to the expected output shape by removing the last dimension
    # New shape: (batch, channel, new_height, new_width)
    result = pooled_input.view(batch, channel, new_h_dim, new_w_dim)

    # Return the result tensor after pooling
    return result


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Randomly drop out elements of the input tensor with probability rate.
    
    Args:
    ----
        input: Input tensor
        p: Dropout probability between 0 and 1
        ignore: Whether to ignore dropout
        
    Returns:
    -------
        Output with random values dropped to 0

    """
    # If ignore is true or p is 0, return the input as is
    if ignore or p <= 0:
        return input
    
    # If p is 1, return a tensor of zeros with the same shape as the input
    if p >= 1:
        return input.zeros(input.shape)        

    # Generate random mask and drop out elements which the corresponding mask is less than p
    mask = rand(input.shape) > p

    # Return the input tensor with the dropped out elements
    return input * mask
