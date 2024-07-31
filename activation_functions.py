from torch import Size, Tensor
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

# Maxpool activation function. Isolates the largest entry per kernel_size * kernel_size. Stride = 2 halves the output tensor height and width.
# https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
def max_pool2d(x: Tensor, kernel_size: int = 2, stride: int = 2):    
    n_rows = ((x.size(2) - (kernel_size - 1) ) // stride) + 1
    n_columns = ((x.size(3) - (kernel_size - 1) ) // stride) + 1

    output = torch.empty(Size((x.size(0), x.size(1), n_rows, n_columns)), device=x.device)

    for batch_index in range(x.size(0)):
        for features_index in range(x.size(1)):
            features = x[batch_index, features_index]

            for row in range(n_rows):
                for column in range(n_columns):
                    feature_row_index = row * stride
                    feature_column_index = column * stride
                    output[batch_index, features_index, row, column] = features[feature_row_index:feature_row_index + kernel_size, feature_column_index:feature_column_index + kernel_size].max()

    return output