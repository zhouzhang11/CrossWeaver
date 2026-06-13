"""
DCN (Deformable Convolution) 相关模块
用于 FPN 中的跨模态空间对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "DCN_Offset_Conv",
    "DCN_Warp_RGB",
    "DCN_Warp_IR",
)


class DCN_Offset_Conv(nn.Module):
    """学习 DCN offset 的卷积层"""
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size
        
        # 输入是 tuple (RGB, IR) 拼接后的特征
        # 输出是 2 * kernel_size^2 个 offset (x, y 方向)
        self.offset_conv = nn.Conv2d(
            c * 2,  # 输入通道: RGB + IR
            2 * kernel_size * kernel_size,  # 输出: offset 数量
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        # 初始化为 0, 即初始时无偏移
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
    
    def forward(self, x):
        """
        输入: (RGB, IR) tuple 或 list[(RGB, IR)]
        输出: offset tensor [B, 2*K*K, H, W]
        """
        # 处理输入格式
        if isinstance(x, list):
            # 如果是 list, 取第一个元素
            x = x[0]
        
        if isinstance(x, tuple):
            # 拼接 RGB 和 IR
            x_cat = torch.cat([x[0], x[1]], dim=1)
        else:
            # 如果已经是拼接好的张量
            x_cat = x
        
        offset = self.offset_conv(x_cat)
        return offset


class DCN_Warp_RGB(nn.Module):
    """对 RGB 分支使用负 offset 进行可变形卷积"""
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size
    
    def forward(self, x):
        """
        输入: [tuple_features, offset]
        - tuple_features: (RGB, IR) tuple
        - offset: [B, 2*K*K, H, W]
        输出: warped RGB tensor
        """
        if isinstance(x[0], tuple):
            rgb_feat = x[0][0]  # 从 tuple 取 RGB
        else:
            rgb_feat = x[0]
        
        offset = x[1]
        
        # 使用负 offset (对称)
        warped = self._grid_sample_with_offset(rgb_feat, -offset)
        return warped
    
    def _grid_sample_with_offset(self, feat, offset):
        """使用 offset 进行 grid sampling"""
        B, C, H, W = feat.shape
        
        # 简化实现: 使用双线性插值
        # 创建基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=feat.dtype),
            torch.arange(W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = grid.repeat(B, 1, 1, 1)  # [B, 2, H, W]
        
        # 添加 offset (简化: 只使用前两个通道作为 x, y offset)
        offset_xy = offset[:, :2, :, :]  # [B, 2, H, W]
        grid = grid + offset_xy
        
        # 归一化到 [-1, 1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0
        
        # grid_sample 需要 [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # 双线性插值
        warped = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped


class DCN_Warp_IR(nn.Module):
    """对 IR 分支使用正 offset 进行可变形卷积"""
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size
    
    def forward(self, x):
        """
        输入: [tuple_features, offset]
        - tuple_features: (RGB, IR) tuple
        - offset: [B, 2*K*K, H, W]
        输出: warped IR tensor
        """
        if isinstance(x[0], tuple):
            ir_feat = x[0][1]  # 从 tuple 取 IR
        else:
            ir_feat = x[0]
        
        offset = x[1]
        
        # 使用正 offset
        warped = self._grid_sample_with_offset(ir_feat, offset)
        return warped
    
    def _grid_sample_with_offset(self, feat, offset):
        """使用 offset 进行 grid sampling"""
        B, C, H, W = feat.shape
        
        # 简化实现: 使用双线性插值
        # 创建基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=feat.dtype),
            torch.arange(W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = grid.repeat(B, 1, 1, 1)  # [B, 2, H, W]
        
        # 添加 offset (简化: 只使用前两个通道作为 x, y offset)
        offset_xy = offset[:, :2, :, :]  # [B, 2, H, W]
        grid = grid + offset_xy
        
        # 归一化到 [-1, 1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0
        
        # grid_sample 需要 [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # 双线性插值
        warped = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
