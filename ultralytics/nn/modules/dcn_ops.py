"""
DCN (Deformable Convolution) modules
Used for cross-modal spatial alignment in FPN
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
    """Learnable DCN offset convolution layer"""
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size

        # Input is concatenated (RGB, IR) features
        # Output is 2 * kernel_size^2 offsets (x, y directions)
        self.offset_conv = nn.Conv2d(
            c * 2,  # Input channels: RGB + IR
            2 * kernel_size * kernel_size,  # Output: number of offsets
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # Initialize to 0 so initial offsets are zero
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        """
        Input: (RGB, IR) tuple or list[(RGB, IR)]
        Output: offset tensor [B, 2*K*K, H, W]
        """
        # Handle input format
        if isinstance(x, list):
            # If list, take first element
            x = x[0]

        if isinstance(x, tuple):
            # Concatenate RGB and IR
            x_cat = torch.cat([x[0], x[1]], dim=1)
        else:
            # If already concatenated tensor
            x_cat = x

        offset = self.offset_conv(x_cat)
        return offset


class DCN_Warp_RGB(nn.Module):
    """Apply deformable convolution with negative offset on RGB branch"""
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Input: [tuple_features, offset]
        - tuple_features: (RGB, IR) tuple
        - offset: [B, 2*K*K, H, W]
        Output: warped RGB tensor
        """
        if isinstance(x[0], tuple):
            rgb_feat = x[0][0]  # Extract RGB from tuple
        else:
            rgb_feat = x[0]

        offset = x[1]

        # Use negative offset (symmetric)
        warped = self._grid_sample_with_offset(rgb_feat, -offset)
        return warped

    def _grid_sample_with_offset(self, feat, offset):
        """Apply grid sampling with offset"""
        B, C, H, W = feat.shape

        # Simplified implementation: use bilinear interpolation
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=feat.dtype),
            torch.arange(W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = grid.repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # Add offset (simplified: use first two channels as x, y offset)
        offset_xy = offset[:, :2, :, :]  # [B, 2, H, W]
        grid = grid + offset_xy

        # Normalize to [-1, 1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0

        # grid_sample expects [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Bilinear interpolation
        warped = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped


class DCN_Warp_IR(nn.Module):
    """Apply deformable convolution with positive offset on IR branch"""
    def __init__(self, c, kernel_size=3):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        Input: [tuple_features, offset]
        - tuple_features: (RGB, IR) tuple
        - offset: [B, 2*K*K, H, W]
        Output: warped IR tensor
        """
        if isinstance(x[0], tuple):
            ir_feat = x[0][1]  # Extract IR from tuple
        else:
            ir_feat = x[0]

        offset = x[1]

        # Use positive offset
        warped = self._grid_sample_with_offset(ir_feat, offset)
        return warped

    def _grid_sample_with_offset(self, feat, offset):
        """Apply grid sampling with offset"""
        B, C, H, W = feat.shape

        # Simplified implementation: use bilinear interpolation
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=feat.dtype),
            torch.arange(W, device=feat.device, dtype=feat.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        grid = grid.repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # Add offset (simplified: use first two channels as x, y offset)
        offset_xy = offset[:, :2, :, :]  # [B, 2, H, W]
        grid = grid + offset_xy

        # Normalize to [-1, 1]
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0

        # grid_sample expects [B, H, W, 2]
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Bilinear interpolation
        warped = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return warped
