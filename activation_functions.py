from torch import Tensor
import torch
import torch.nn.functional as F

def relu(x: Tensor):
    # If a given input is larger than 0, return x otherwise return 0.
    return torch.where(x > 0, x, 0.)