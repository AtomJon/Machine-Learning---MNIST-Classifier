from torch import Tensor
import torch
import torch.nn.functional as F

# ReLu activation function. Disables all values in an arbitrary tensor that are less than 0
def relu(x: Tensor):
    # If a given input is larger than 0, return x otherwise return 0.
    return torch.where(x > 0, x, 0.)