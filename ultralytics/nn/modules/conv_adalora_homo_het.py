"""
Homogeneous and heterogeneous feature separation and extraction modules
Based on AdaLoRA symmetric dual-modal architecture
"""

import torch
import torch.nn as nn
import math

__all__ = (
    "Conv_adalora_sym_m_homo",
    "C2f_adalora_sym_m_homo",
    "SPPF_adalora_sym_m_homo",
    "Conv_adalora_sym_m_het",
    "C2f_adalora_sym_m_het",
    "SPPF_adalora_sym_m_het",
    "PriorExtractor_Multi",
    "Add_Prior",
)


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ============ Homogeneous Feature Base Classes ============

class Conv2d_adalora_base_homo(nn.Module):
    """Homogeneous feature AdaLoRA convolution base: shared weights only"""
    def __init__(self, c1, c2, k, r, lora_alpha, lora_dropout, stride, padding, groups, dilation, bias):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Only initialize shared weights
        self.conv = nn.Conv2d(c1, c2, k, stride, padding, dilation, groups, bias)

    def reset_parameters(self):
        self.conv.reset_parameters()


class Conv2d_adalora_homo(Conv2d_adalora_base_homo):
    """Extract homogeneous features: shared weights only"""
    def forward(self, x):
        """
        Input: (x_rgb, x_ir) tuple
        Output: (homo_rgb, homo_ir) tuple
        """
        x_rgb = self.conv._conv_forward(x[0], self.conv.weight, self.conv.bias)
        x_ir = self.conv._conv_forward(x[1], self.conv.weight, self.conv.bias)
        return (x_rgb, x_ir)


# ============ Heterogeneous Feature Base Classes ============

class Conv2d_adalora_base_het(nn.Module):
    """Heterogeneous feature AdaLoRA convolution base: LoRA parameters only"""
    def __init__(self, c1, c2, k, r, lora_alpha, lora_dropout, stride, padding, groups, dilation, bias):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.c1 = c1
        self.c2 = c2
        self.kernel_size = k if isinstance(k, int) else k[0]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Only initialize LoRA parameters, no shared weights
        if r > 0:
            # LoRA decomposition: W [c2, c1, k, k] -> [c2, c1*k*k] = B @ A
            # A: [r, c1*k*k]
            # B: [c2, r]
            self.lora_A_rgb = nn.Parameter(torch.zeros((r, c1 * self.kernel_size * self.kernel_size)))
            self.lora_A_ir = nn.Parameter(torch.zeros((r, c1 * self.kernel_size * self.kernel_size)))
            self.lora_E = nn.Parameter(torch.zeros((r, 1)))
            self.lora_B_rgb = nn.Parameter(torch.zeros((c2, r)))
            self.lora_B_ir = nn.Parameter(torch.zeros((c2, r)))

            self.ranknum = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha / self.r
            self.ranknum.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A_rgb'):
            nn.init.kaiming_uniform_(self.lora_A_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_ir, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_ir, a=math.sqrt(5))
            nn.init.zeros_(self.lora_E)


class Conv2d_adalora_het(Conv2d_adalora_base_het):
    """Extract heterogeneous features: LoRA increments only"""
    def forward(self, x):
        if self.r > 0:
            # lora_B_rgb: [c2, r]
            # lora_A_rgb: [r, c1*k*k]
            # lora_E: [r, 1]
            # Result: [c2, r] @ [r, c1*k*k] = [c2, c1*k*k]
            delta_rgb_flat = (self.lora_B_rgb @ (self.lora_A_rgb * self.lora_E))  # [c2, c1*k*k]
            delta_rgb = delta_rgb_flat.view(self.c2, self.c1, self.kernel_size, self.kernel_size) * self.scaling

            delta_ir_flat = (self.lora_B_ir @ (self.lora_A_ir * self.lora_E))
            delta_ir = delta_ir_flat.view(self.c2, self.c1, self.kernel_size, self.kernel_size) * self.scaling

            # Apply via F.conv2d
            x_rgb = torch.nn.functional.conv2d(x[0], delta_rgb, None, self.stride, self.padding, self.dilation, self.groups)
            x_ir = torch.nn.functional.conv2d(x[1], delta_ir, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            # If r=0, return zero tensors
            B, C, H, W = x[0].shape
            out_h = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
            out_w = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
            x_rgb = torch.zeros(B, self.c2, out_h, out_w, device=x[0].device, dtype=x[0].dtype)
            x_ir = torch.zeros(B, self.c2, out_h, out_w, device=x[1].device, dtype=x[1].dtype)

        return (x_rgb, x_ir)


# ============ Homogeneous Feature Extraction Modules ============

class Conv_adalora_sym_m_homo(nn.Module):
    """Homogeneous feature extraction convolution layer"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, r=0, lora_alpha=1, lora_dropout=0., p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv2d_adalora_homo(c1, c2, k, r, lora_alpha, lora_dropout,
                                        stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn_rgb = nn.BatchNorm2d(c2)
        self.bn_ir = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Input/Output: (rgb, ir) tuple"""
        x = self.conv(x)
        x_rgb = self.act(self.bn_rgb(x[0]))
        x_ir = self.act(self.bn_ir(x[1]))
        return (x_rgb, x_ir)


class Bottleneck_adalora_homo(nn.Module):
    """Homogeneous feature Bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv_adalora_sym_m_homo(c1, c_, k[0], 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m_homo(c_, c2, k[1], 1, r, lora_alpha, lora_dropout, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        if self.add:
            x_1 = self.cv2(self.cv1(x))
            return (x[0] + x_1[0], x[1] + x_1[1])
        else:
            return self.cv2(self.cv1(x))


class C2f_adalora_sym_m_homo(nn.Module):
    """Homogeneous feature C2f"""
    def __init__(self, c1, c2, n=1, shortcut=False, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv_adalora_sym_m_homo(c1, 2 * self.c, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m_homo((2 + n) * self.c, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.m = nn.ModuleList(Bottleneck_adalora_homo(self.c, self.c, shortcut, g, k=(3, 3), e=1.0,
                                                        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                               for _ in range(n))

    def forward(self, x):
        x = self.cv1(x)
        y_rgb = list(x[0].chunk(2, 1))
        y_ir = list(x[1].chunk(2, 1))
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
        y_rgb = torch.cat(y_rgb, 1)
        y_ir = torch.cat(y_ir, 1)
        return self.cv2((y_rgb, y_ir))


class SPPF_adalora_sym_m_homo(nn.Module):
    """Homogeneous feature SPPF"""
    def __init__(self, c1, c2, k=5, r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv_adalora_sym_m_homo(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m_homo(c_ * 4, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1_rgb = self.m(x[0])
        y1_ir = self.m(x[1])
        y2_rgb = self.m(y1_rgb)
        y2_ir = self.m(y1_ir)
        return self.cv2((torch.cat((x[0], y1_rgb, y2_rgb, self.m(y2_rgb)), 1),
                         torch.cat((x[1], y1_ir, y2_ir, self.m(y2_ir)), 1)))


# ============ Heterogeneous Feature Extraction Modules ============

class Conv_adalora_sym_m_het(nn.Module):
    """Heterogeneous feature extraction convolution layer"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, r=0, lora_alpha=1, lora_dropout=0., p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv_het = Conv2d_adalora_het(c1, c2, k, r, lora_alpha, lora_dropout,
                                       stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn_rgb = nn.BatchNorm2d(c2)
        self.bn_ir = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Input/Output: (rgb, ir) tuple"""
        x = self.conv_het(x)
        x_rgb = self.act(self.bn_rgb(x[0]))
        x_ir = self.act(self.bn_ir(x[1]))
        return (x_rgb, x_ir)


class Bottleneck_adalora_het(nn.Module):
    """Heterogeneous feature Bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv_adalora_sym_m_het(c1, c_, k[0], 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m_het(c_, c2, k[1], 1, r, lora_alpha, lora_dropout, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        if self.add:
            x_1 = self.cv2(self.cv1(x))
            return (x[0] + x_1[0], x[1] + x_1[1])
        else:
            return self.cv2(self.cv1(x))


class C2f_adalora_sym_m_het(nn.Module):
    """Heterogeneous feature C2f"""
    def __init__(self, c1, c2, n=1, shortcut=False, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv_adalora_sym_m_het(c1, 2 * self.c, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m_het((2 + n) * self.c, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.m = nn.ModuleList(Bottleneck_adalora_het(self.c, self.c, shortcut, g, k=(3, 3), e=1.0,
                                                       r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                               for _ in range(n))

    def forward(self, x):
        x = self.cv1(x)
        y_rgb = list(x[0].chunk(2, 1))
        y_ir = list(x[1].chunk(2, 1))
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
        y_rgb = torch.cat(y_rgb, 1)
        y_ir = torch.cat(y_ir, 1)
        return self.cv2((y_rgb, y_ir))


class SPPF_adalora_sym_m_het(nn.Module):
    """Heterogeneous feature SPPF"""
    def __init__(self, c1, c2, k=5, r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv_adalora_sym_m_het(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m_het(c_ * 4, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1_rgb = self.m(x[0])
        y1_ir = self.m(x[1])
        y2_rgb = self.m(y1_rgb)
        y2_ir = self.m(y1_ir)
        return self.cv2((torch.cat((x[0], y1_rgb, y2_rgb, self.m(y2_rgb)), 1),
                         torch.cat((x[1], y1_ir, y2_ir, self.m(y2_ir)), 1)))


# ============ Prior Extraction and Injection Modules ============

class PriorExtractor_Multi(nn.Module):
    """
    Multi-scale prior extraction module
    Input: (RGB, IR) tuple, each [B, 3, H, W]
    Output: three-scale prior tuples:
        (
          (Prior_P3_RGB, Prior_P3_IR),
          (Prior_P4_RGB, Prior_P4_IR),
          (Prior_P5_RGB, Prior_P5_IR)
        )
    """
    def __init__(self, c3=256, c4=512, c5=1024, sigma=0.05, window_size=5):
        super().__init__()
        self.sigma = sigma
        self.window_size = window_size

        # Fixed convolution kernels (non-trainable)
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Laplacian operator
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian)

        # Trainable translation layers (1x1 conv)
        # RGB prior translation (1 channel -> c3/c4/c5)
        self.rgb_translator_p3 = nn.Conv2d(1, c3, 1, bias=False)
        self.rgb_translator_p4 = nn.Conv2d(1, c4, 1, bias=False)
        self.rgb_translator_p5 = nn.Conv2d(1, c5, 1, bias=False)

        # IR prior translation (2 channels -> c3/c4/c5)
        self.ir_translator_p3 = nn.Conv2d(2, c3, 1, bias=False)
        self.ir_translator_p4 = nn.Conv2d(2, c4, 1, bias=False)
        self.ir_translator_p5 = nn.Conv2d(2, c5, 1, bias=False)

        # Initialize translation layers
        for m in [self.rgb_translator_p3, self.rgb_translator_p4, self.rgb_translator_p5,
                  self.ir_translator_p3, self.ir_translator_p4, self.ir_translator_p5]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def rgb_to_gray(self, rgb):
        """RGB to grayscale [B,3,H,W] -> [B,1,H,W]"""
        # Standard weights: 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]
        return gray

    def extract_rgb_prior(self, rgb):
        """Extract RGB prior: Laplacian texture [B,3,H,W] -> [B,1,H,W]"""
        gray = self.rgb_to_gray(rgb)
        # Apply Laplacian operator
        laplacian_out = torch.nn.functional.conv2d(gray, self.laplacian, padding=1)
        # Take absolute value
        prior = torch.abs(laplacian_out)
        return prior

    def extract_ir_prior(self, ir):
        """Extract IR prior: Sobel gradient + thermal continuity [B,3,H,W] -> [B,2,H,W]"""
        gray = self.rgb_to_gray(ir)

        # 1. Sobel gradient prior
        gx = torch.nn.functional.conv2d(gray, self.sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(gray, self.sobel_y, padding=1)
        grad_prior = torch.abs(gx) + torch.abs(gy)  # [B,1,H,W]

        # 2. Thermal continuity prior (based on local variance)
        # Extract local windows using unfold
        B, C, H, W = gray.shape
        pad = self.window_size // 2
        gray_padded = torch.nn.functional.pad(gray, (pad, pad, pad, pad), mode='reflect')

        # unfold: [B, 1, H, W] -> [B, 1*window_size*window_size, H*W]
        patches = torch.nn.functional.unfold(gray_padded, kernel_size=self.window_size, stride=1)
        patches = patches.view(B, self.window_size * self.window_size, H, W)

        # Compute local variance
        mean = patches.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        var = ((patches - mean) ** 2).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Thermal continuity: exp(-var/sigma^2)
        cont_prior = torch.exp(-var / (self.sigma ** 2))  # [B,1,H,W]

        # Concatenate both priors
        ir_prior = torch.cat([grad_prior, cont_prior], dim=1)  # [B,2,H,W]
        return ir_prior

    def forward(self, x):
        """
        Input: (RGB, IR) tuple
        Output: ((P3_RGB, P3_IR), (P4_RGB, P4_IR), (P5_RGB, P5_IR))
        """
        rgb, ir = x

        # Stage 1: Extract raw priors
        P_RGB_raw = self.extract_rgb_prior(rgb)  # [B,1,H,W]
        P_IR_raw = self.extract_ir_prior(ir)      # [B,2,H,W]

        # Stage 2: Multi-scale translation
        # P3 level (8x downsampling)
        P_RGB_p3_down = torch.nn.functional.avg_pool2d(P_RGB_raw, kernel_size=8, stride=8)
        P_IR_p3_down = torch.nn.functional.avg_pool2d(P_IR_raw, kernel_size=8, stride=8)
        Prior_P3_RGB = self.rgb_translator_p3(P_RGB_p3_down)
        Prior_P3_IR = self.ir_translator_p3(P_IR_p3_down)

        # P4 level (16x downsampling)
        P_RGB_p4_down = torch.nn.functional.avg_pool2d(P_RGB_raw, kernel_size=16, stride=16)
        P_IR_p4_down = torch.nn.functional.avg_pool2d(P_IR_raw, kernel_size=16, stride=16)
        Prior_P4_RGB = self.rgb_translator_p4(P_RGB_p4_down)
        Prior_P4_IR = self.ir_translator_p4(P_IR_p4_down)

        # P5 level (32x downsampling)
        P_RGB_p5_down = torch.nn.functional.avg_pool2d(P_RGB_raw, kernel_size=32, stride=32)
        P_IR_p5_down = torch.nn.functional.avg_pool2d(P_IR_raw, kernel_size=32, stride=32)
        Prior_P5_RGB = self.rgb_translator_p5(P_RGB_p5_down)
        Prior_P5_IR = self.ir_translator_p5(P_IR_p5_down)

        return (
            (Prior_P3_RGB, Prior_P3_IR),
            (Prior_P4_RGB, Prior_P4_IR),
            (Prior_P5_RGB, Prior_P5_IR)
        )


class Add_Prior(nn.Module):
    """
    Prior injection module: add the corresponding scale prior to heterogeneous features
    Input: [F_Het_tuple, Prior_Multi_tuple, scale_index]
    Output: (RGB_enhanced, IR_enhanced) tuple
    """
    def __init__(self, c, scale_index):
        super().__init__()
        self.c = c
        self.scale_index = scale_index  # 0=P3, 1=P4, 2=P5

    def forward(self, x):
        """
        x: [F_Het_tuple, Prior_Multi_tuple]
        F_Het_tuple: (RGB, IR) heterogeneous features
        Prior_Multi_tuple: ((P3_RGB, P3_IR), (P4_RGB, P4_IR), (P5_RGB, P5_IR))
        """
        f_het = x[0]  # (RGB, IR) tuple
        prior_multi = x[1]  # Three-scale priors

        # Extract the corresponding scale prior
        prior_scale = prior_multi[self.scale_index]  # (Prior_RGB, Prior_IR)

        # Add for enhancement
        rgb_enhanced = f_het[0] + prior_scale[0]
        ir_enhanced = f_het[1] + prior_scale[1]

        return (rgb_enhanced, ir_enhanced)
