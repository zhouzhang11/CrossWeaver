"""
Tuple 操作模块: 用于处理 (RGB, IR) tuple 的 Add 和 Concat 操作
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
    """从 tuple 中提取 RGB 分支并相加"""
    def __init__(self, c):
        super().__init__()
        self.c = c
    
    def forward(self, x):
        """
        输入: [tensor1, tuple2] 或 [tensor1, tensor2]
        - tensor1: 单个张量 (如 FPN_P3_RGB)
        - tuple2: (RGB, IR) tuple (如 F_Het from L14) 或单个张量
        输出: tensor1 + tuple2[0] (如果是 tuple) 或 tensor1 + tuple2
        """
        if isinstance(x[1], tuple):
            return x[0] + x[1][0]  # 取 tuple 的第一个元素 (RGB)
        else:
            return x[0] + x[1]


class Add_IR(nn.Module):
    """从 tuple 中提取 IR 分支并相加"""
    def __init__(self, c):
        super().__init__()
        self.c = c
    
    def forward(self, x):
        """
        输入: [tensor1, tuple2] 或 [tensor1, tensor2]
        - tensor1: 单个张量 (如 FPN_P3_IR)
        - tuple2: (RGB, IR) tuple (如 F_Het from L14) 或单个张量
        输出: tensor1 + tuple2[1] (如果是 tuple) 或 tensor1 + tuple2
        """
        if isinstance(x[1], tuple):
            return x[0] + x[1][1]  # 取 tuple 的第二个元素 (IR)
        else:
            return x[0] + x[1]


class Concat_RGB(nn.Module):
    """从 tuple 中提取 RGB 分支并拼接"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        """
        输入: list of tensors/tuples
        输出: 拼接所有 RGB 分支
        """
        rgb_tensors = []
        for item in x:
            if isinstance(item, tuple):
                rgb_tensors.append(item[0])  # 取 RGB 分支
            else:
                rgb_tensors.append(item)  # 已经是单个张量
        return torch.cat(rgb_tensors, self.d)


class Concat_IR(nn.Module):
    """从 tuple 中提取 IR 分支并拼接"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        """
        输入: list of tensors/tuples
        输出: 拼接所有 IR 分支
        """
        ir_tensors = []
        for item in x:
            if isinstance(item, tuple):
                ir_tensors.append(item[1])  # 取 IR 分支
            else:
                ir_tensors.append(item)  # 已经是单个张量
        return torch.cat(ir_tensors, self.d)
