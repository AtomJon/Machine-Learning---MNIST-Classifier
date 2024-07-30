from torch import Tensor
import torch
import torch.nn.functional as F

# ReLu activation function. Disables all values in an arbitrary tensor that are less than 0
def relu(x: Tensor):
    # If a given input is larger than 0, return x otherwise return 0
    return torch.where(x > 0, x, 0.)

# Dropout activation function for disabling random entries with p probability. This prevents co-adaptation of feature-detectors and reduces over-training
def dropout(x: Tensor, training: bool, p = 0.5):
    # Only apply dropout while training
    if training == False:
        return x
    
    # Either identity or 0-matrix with *p* probability of a 0 in each entry.
    activation_matrix = (torch.rand(x.shape, device=x.device) > p)

    # Apply acitvation matrix and return
    return activation_matrix * x