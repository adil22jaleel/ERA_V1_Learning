import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

torch.manual_seed(1)

def print_summary(model, input_size=(3, 32, 32)):
    """Print Model summary

    Args:
        model (Net): Model Instance
        input_size (tuple, optional): Input size. Defaults to (1, 28, 28).
    """
    summary(model, input_size=input_size)