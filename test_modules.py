"""
测试同质/异质特征提取模块和相关操作模块
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from ultralytics.nn.modules.conv_adalora_homo_het import (
    Conv_adalora_sym_m_homo,
    Conv_adalora_sym_m_het,
    C2f_adalora_sym_m_homo,
    C2f_adalora_sym_m_het,
)
from ultralytics.nn.modules.tuple_ops import (
    Add_RGB,
    Add_IR,
    Concat_RGB,
    Concat_IR,
)
from ultralytics.nn.modules.dcn_ops import (
    DCN_Offset_Conv,
    DCN_Warp_RGB,
    DCN_Warp_IR,
)


def test_homo_het_conv():
    """测试同质/异质特征提取卷积"""
    print("\n" + "="*60)
    print("测试 1: 同质/异质特征提取卷积")
    print("="*60)
    
    batch_size = 2
    c_in = 64
    c_out = 128
    h, w = 32, 32
    
    # 创建输入
    x_rgb = torch.randn(batch_size, c_in, h, w)
    x_ir = torch.randn(batch_size, c_in, h, w)
    x = (x_rgb, x_ir)
    
    # 测试同质特征提取
    conv_homo = Conv_adalora_sym_m_homo(c_in, c_out, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)
    out_homo = conv_homo(x)
    
    print(f"输入 RGB shape: {x_rgb.shape}")
    print(f"输入 IR shape: {x_ir.shape}")
    print(f"同质特征输出 RGB shape: {out_homo[0].shape}")
    print(f"同质特征输出 IR shape: {out_homo[1].shape}")
    assert out_homo[0].shape == (batch_size, c_out, h, w), "同质特征 RGB 输出维度错误"
    assert out_homo[1].shape == (batch_size, c_out, h, w), "同质特征 IR 输出维度错误"
    print("✓ 同质特征提取测试通过")
    
    # 测试异质特征提取
    conv_het = Conv_adalora_sym_m_het(c_in, c_out, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)
    out_het = conv_het(x)
    
    print(f"异质特征输出 RGB shape: {out_het[0].shape}")
    print(f"异质特征输出 IR shape: {out_het[1].shape}")
    assert out_het[0].shape == (batch_size, c_out, h, w), "异质特征 RGB 输出维度错误"
    assert out_het[1].shape == (batch_size, c_out, h, w), "异质特征 IR 输出维度错误"
    print("✓ 异质特征提取测试通过")


def test_homo_het_c2f():
    """测试同质/异质特征提取 C2f"""
    print("\n" + "="*60)
    print("测试 2: 同质/异质特征提取 C2f")
    print("="*60)
    
    batch_size = 2
    c_in = 128
    c_out = 256
    h, w = 16, 16
    
    # 创建输入
    x_rgb = torch.randn(batch_size, c_in, h, w)
    x_ir = torch.randn(batch_size, c_in, h, w)
    x = (x_rgb, x_ir)
    
    # 测试同质特征 C2f
    c2f_homo = C2f_adalora_sym_m_homo(c_in, c_out, n=3, shortcut=True, r=9, lora_alpha=9, lora_dropout=0.)
    out_homo = c2f_homo(x)
    
    print(f"输入 RGB shape: {x_rgb.shape}")
    print(f"输入 IR shape: {x_ir.shape}")
    print(f"C2f 同质特征输出 RGB shape: {out_homo[0].shape}")
    print(f"C2f 同质特征输出 IR shape: {out_homo[1].shape}")
    assert out_homo[0].shape == (batch_size, c_out, h, w), "C2f 同质特征 RGB 输出维度错误"
    assert out_homo[1].shape == (batch_size, c_out, h, w), "C2f 同质特征 IR 输出维度错误"
    print("✓ C2f 同质特征提取测试通过")
    
    # 测试异质特征 C2f
    c2f_het = C2f_adalora_sym_m_het(c_in, c_out, n=3, shortcut=True, r=9, lora_alpha=9, lora_dropout=0.)
    out_het = c2f_het(x)
    
    print(f"C2f 异质特征输出 RGB shape: {out_het[0].shape}")
    print(f"C2f 异质特征输出 IR shape: {out_het[1].shape}")
    assert out_het[0].shape == (batch_size, c_out, h, w), "C2f 异质特征 RGB 输出维度错误"
    assert out_het[1].shape == (batch_size, c_out, h, w), "C2f 异质特征 IR 输出维度错误"
    print("✓ C2f 异质特征提取测试通过")


def test_add_rgb_ir():
    """测试 Add_RGB 和 Add_IR"""
    print("\n" + "="*60)
    print("测试 3: Add_RGB 和 Add_IR")
    print("="*60)
    
    batch_size = 2
    c = 256
    h, w = 16, 16
    
    # 创建输入
    fpn_rgb = torch.randn(batch_size, c, h, w)
    fpn_ir = torch.randn(batch_size, c, h, w)
    het_rgb = torch.randn(batch_size, c, h, w)
    het_ir = torch.randn(batch_size, c, h, w)
    het_tuple = (het_rgb, het_ir)
    
    # 测试 Add_RGB
    add_rgb = Add_RGB(c)
    out_rgb = add_rgb([fpn_rgb, het_tuple])
    
    print(f"FPN RGB shape: {fpn_rgb.shape}")
    print(f"Het tuple RGB shape: {het_rgb.shape}")
    print(f"Add_RGB 输出 shape: {out_rgb.shape}")
    assert out_rgb.shape == (batch_size, c, h, w), "Add_RGB 输出维度错误"
    print("✓ Add_RGB 测试通过")
    
    # 测试 Add_IR
    add_ir = Add_IR(c)
    out_ir = add_ir([fpn_ir, het_tuple])
    
    print(f"FPN IR shape: {fpn_ir.shape}")
    print(f"Het tuple IR shape: {het_ir.shape}")
    print(f"Add_IR 输出 shape: {out_ir.shape}")
    assert out_ir.shape == (batch_size, c, h, w), "Add_IR 输出维度错误"
    print("✓ Add_IR 测试通过")


def test_concat_rgb_ir():
    """测试 Concat_RGB 和 Concat_IR"""
    print("\n" + "="*60)
    print("测试 4: Concat_RGB 和 Concat_IR")
    print("="*60)
    
    batch_size = 2
    c = 256
    h, w = 16, 16
    
    # 创建输入
    feat1 = torch.randn(batch_size, c, h, w)
    feat2_rgb = torch.randn(batch_size, c, h, w)
    feat2_ir = torch.randn(batch_size, c, h, w)
    feat2_tuple = (feat2_rgb, feat2_ir)
    feat3_rgb = torch.randn(batch_size, c, h, w)
    feat3_ir = torch.randn(batch_size, c, h, w)
    feat3_tuple = (feat3_rgb, feat3_ir)
    
    # 测试 Concat_RGB
    concat_rgb = Concat_RGB(dimension=1)
    out_rgb = concat_rgb([feat1, feat2_tuple, feat3_tuple])
    
    print(f"Feat1 shape: {feat1.shape}")
    print(f"Feat2 tuple RGB shape: {feat2_rgb.shape}")
    print(f"Feat3 tuple RGB shape: {feat3_rgb.shape}")
    print(f"Concat_RGB 输出 shape: {out_rgb.shape}")
    assert out_rgb.shape == (batch_size, c * 3, h, w), "Concat_RGB 输出维度错误"
    print("✓ Concat_RGB 测试通过")
    
    # 测试 Concat_IR
    concat_ir = Concat_IR(dimension=1)
    out_ir = concat_ir([feat1, feat2_tuple, feat3_tuple])
    
    print(f"Concat_IR 输出 shape: {out_ir.shape}")
    assert out_ir.shape == (batch_size, c * 3, h, w), "Concat_IR 输出维度错误"
    print("✓ Concat_IR 测试通过")


def test_dcn_modules():
    """测试 DCN 相关模块"""
    print("\n" + "="*60)
    print("测试 5: DCN 相关模块")
    print("="*60)
    
    batch_size = 2
    c = 512
    h, w = 16, 16
    
    # 创建输入
    feat_rgb = torch.randn(batch_size, c, h, w)
    feat_ir = torch.randn(batch_size, c, h, w)
    feat_tuple = (feat_rgb, feat_ir)
    
    # 测试 DCN_Offset_Conv
    dcn_offset = DCN_Offset_Conv(c, kernel_size=3)
    offset = dcn_offset(feat_tuple)
    
    print(f"输入 RGB shape: {feat_rgb.shape}")
    print(f"输入 IR shape: {feat_ir.shape}")
    print(f"DCN Offset 输出 shape: {offset.shape}")
    assert offset.shape[0] == batch_size, "DCN Offset batch 维度错误"
    assert offset.shape[1] == 2 * 3 * 3, "DCN Offset 通道数错误"
    assert offset.shape[2:] == (h, w), "DCN Offset 空间维度错误"
    print("✓ DCN_Offset_Conv 测试通过")
    
    # 测试 DCN_Warp_RGB
    dcn_warp_rgb = DCN_Warp_RGB(c, kernel_size=3)
    warped_rgb = dcn_warp_rgb([feat_tuple, offset])
    
    print(f"DCN Warp RGB 输出 shape: {warped_rgb.shape}")
    assert warped_rgb.shape == (batch_size, c, h, w), "DCN Warp RGB 输出维度错误"
    print("✓ DCN_Warp_RGB 测试通过")
    
    # 测试 DCN_Warp_IR
    dcn_warp_ir = DCN_Warp_IR(c, kernel_size=3)
    warped_ir = dcn_warp_ir([feat_tuple, offset])
    
    print(f"DCN Warp IR 输出 shape: {warped_ir.shape}")
    assert warped_ir.shape == (batch_size, c, h, w), "DCN Warp IR 输出维度错误"
    print("✓ DCN_Warp_IR 测试通过")


def test_full_pipeline():
    """测试完整的特征流动管道"""
    print("\n" + "="*60)
    print("测试 6: 完整特征流动管道")
    print("="*60)
    
    batch_size = 2
    h, w = 64, 64
    
    # 模拟 Backbone 输出
    print("\n--- Backbone 阶段 ---")
    x_rgb = torch.randn(batch_size, 3, h, w)
    x_ir = torch.randn(batch_size, 3, h, w)
    x = (x_rgb, x_ir)
    
    # P3 级别
    conv_homo_p3 = Conv_adalora_sym_m_homo(3, 256, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)
    conv_het_p3 = Conv_adalora_sym_m_het(3, 256, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)
    
    homo_p3 = conv_homo_p3(x)
    het_p3 = conv_het_p3(x)
    
    print(f"P3 同质特征 RGB shape: {homo_p3[0].shape}")
    print(f"P3 同质特征 IR shape: {homo_p3[1].shape}")
    print(f"P3 异质特征 RGB shape: {het_p3[0].shape}")
    print(f"P3 异质特征 IR shape: {het_p3[1].shape}")
    
    # 模拟 FPN 输出
    print("\n--- FPN 阶段 ---")
    fpn_p3_rgb = torch.randn(batch_size, 256, h, w)
    fpn_p3_ir = torch.randn(batch_size, 256, h, w)
    
    # PAN 阶段: 异质特征注入
    print("\n--- PAN 阶段: 异质特征注入 ---")
    add_rgb = Add_RGB(256)
    add_ir = Add_IR(256)
    
    pan_p3_rgb = add_rgb([fpn_p3_rgb, het_p3])
    pan_p3_ir = add_ir([fpn_p3_ir, het_p3])
    
    print(f"PAN P3 RGB shape: {pan_p3_rgb.shape}")
    print(f"PAN P3 IR shape: {pan_p3_ir.shape}")
    
    assert pan_p3_rgb.shape == (batch_size, 256, h, w), "PAN P3 RGB 维度错误"
    assert pan_p3_ir.shape == (batch_size, 256, h, w), "PAN P3 IR 维度错误"
    
    print("\n✓ 完整管道测试通过")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("开始测试同质/异质特征提取模块")
    print("="*60)
    
    try:
        test_homo_het_conv()
        test_homo_het_c2f()
        test_add_rgb_ir()
        test_concat_rgb_ir()
        test_dcn_modules()
        test_full_pipeline()
        
        print("\n" + "="*60)
        print("✓ 所有测试通过!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
