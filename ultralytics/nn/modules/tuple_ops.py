"""
Tuple operation modules: Add and Concat operations for (RGB, IR) tuples
"""

import torch
import torch.nn as nn

__all__ = (
    "Add_RGB",
    "Add_IR",
    "Concat_RGB",
    "Concat_IR",
)


class Add_RGB(nn.Module):
    """Extract RGB branch from tuple and add"""
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        """
        Input: [tensor1, tuple2] or [tensor1, tensor2]
        - tensor1: single tensor (e.g., FPN_P3_RGB)
        - tuple2: (RGB, IR) tuple (e.g., F_Het from L14) or single tensor
        Output: tensor1 + tuple2[0] (if tuple) or tensor1 + tuple2
        """
        if isinstance(x[1], tuple):
            return x[0] + x[1][0]  # Take first element of tuple (RGB)
        else:
            return x[0] + x[1]


class Add_IR(nn.Module):
    """Extract IR branch from tuple and add"""
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        """
        Input: [tensor1, tuple2] or [tensor1, tensor2]
        - tensor1: single tensor (e.g., FPN_P3_IR)
        - tuple2: (RGB, IR) tuple (e.g., F_Het from L14) or single tensor
        Output: tensor1 + tuple2[1] (if tuple) or tensor1 + tuple2
        """
        if isinstance(x[1], tuple):
            return x[0] + x[1][1]  # Take second element of tuple (IR)
        else:
            return x[0] + x[1]


class Concat_RGB(nn.Module):
    """Extract RGB branch from tuple and concatenate"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Input: list of tensors/tuples
        Output: concatenate all RGB branches
        """
        rgb_tensors = []
        for item in x:
            if isinstance(item, tuple):
                rgb_tensors.append(item[0])  # Take RGB branch
            else:
                rgb_tensors.append(item)  # Already a single tensor
        return torch.cat(rgb_tensors, self.d)


class Concat_IR(nn.Module):
    """Extract IR branch from tuple and concatenate"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Input: list of tensors/tuples
        Output: concatenate all IR branches
        """
        ir_tensors = []
        for item in x:
            if isinstance(item, tuple):
                ir_tensors.append(item[1])  # Take IR branch
            else:
                ir_tensors.append(item)  # Already a single tensor
        return torch.cat(ir_tensors, self.d)
