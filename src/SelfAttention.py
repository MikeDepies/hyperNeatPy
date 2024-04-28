import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """A simple self-attention solution using direct tensor operations."""

    def __init__(self, weights_q: torch.Tensor, bias_q: torch.Tensor, weights_k: torch.Tensor, bias_k: torch.Tensor):
        super(SelfAttention, self).__init__()
        self.data_dim = weights_q.size()[0]
        self.dim_q = weights_q.size()[1]

        # Initialize weights and biases for queries and keys
        self.weights_q = weights_q
        self.bias_q = bias_q

        self.weights_k = weights_k
        self.bias_k = bias_k

    def forward(self, input_data):
        # Expect input_data to be of shape (b, t, k).
        b, t, k = input_data.size()

        # Validate that input dimensions match expected dimensions.
        assert k == self.data_dim, "Input feature dimension does not match data_dim"

        # Compute queries and keys using tensor operations
        queries = torch.matmul(input_data, self.weights_q) + self.bias_q
        keys = torch.matmul(input_data, self.weights_k) + self.bias_k

        # Attention matrix using batch matrix multiplication
        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b, t, t)
        scaled_dot = torch.div(dot, math.sqrt(k))
        return scaled_dot


def calculate_patches(image_height, image_width, patch_height, patch_width, stride_height, stride_width):
    # Calculate number of patches along height
    n_height = (image_height - patch_height) // stride_height + 1
    
    # Calculate number of patches along width
    n_width = (image_width - patch_width) // stride_width + 1
    
    # Handle cases where there is no room for even one patch
    if n_height < 0 or n_width < 0:
        return 0

    # Calculate total number of patches
    total_patches = n_height * n_width
    return total_patches