"""
Tests for homogeneous/heterogeneous feature extraction modules and related operations
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
    """Test homogeneous/heterogeneous feature extraction convolution"""
    print("\n" + "="*60)
    print("Test 1: Homo/Het Feature Extraction Convolution")
    print("="*60)

    batch_size = 2
    c_in = 64
    c_out = 128
    h, w = 32, 32

    # Create input
    x_rgb = torch.randn(batch_size, c_in, h, w)
    x_ir = torch.randn(batch_size, c_in, h, w)
    x = (x_rgb, x_ir)

    # Test homogeneous feature extraction
    conv_homo = Conv_adalora_sym_m_homo(c_in, c_out, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)
    out_homo = conv_homo(x)

    print(f"Input RGB shape: {x_rgb.shape}")
    print(f"Input IR shape: {x_ir.shape}")
    print(f"Homo feature output RGB shape: {out_homo[0].shape}")
    print(f"Homo feature output IR shape: {out_homo[1].shape}")
    assert out_homo[0].shape == (batch_size, c_out, h, w), "Homo feature RGB output dimension error"
    assert out_homo[1].shape == (batch_size, c_out, h, w), "Homo feature IR output dimension error"
    print("PASS: Homo feature extraction test")

    # Test heterogeneous feature extraction
    conv_het = Conv_adalora_sym_m_het(c_in, c_out, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)
    out_het = conv_het(x)

    print(f"Het feature output RGB shape: {out_het[0].shape}")
    print(f"Het feature output IR shape: {out_het[1].shape}")
    assert out_het[0].shape == (batch_size, c_out, h, w), "Het feature RGB output dimension error"
    assert out_het[1].shape == (batch_size, c_out, h, w), "Het feature IR output dimension error"
    print("PASS: Het feature extraction test")


def test_homo_het_c2f():
    """Test homogeneous/heterogeneous feature extraction C2f"""
    print("\n" + "="*60)
    print("Test 2: Homo/Het Feature Extraction C2f")
    print("="*60)

    batch_size = 2
    c_in = 128
    c_out = 256
    h, w = 16, 16

    # Create input
    x_rgb = torch.randn(batch_size, c_in, h, w)
    x_ir = torch.randn(batch_size, c_in, h, w)
    x = (x_rgb, x_ir)

    # Test homogeneous feature C2f
    c2f_homo = C2f_adalora_sym_m_homo(c_in, c_out, n=3, shortcut=True, r=9, lora_alpha=9, lora_dropout=0.)
    out_homo = c2f_homo(x)

    print(f"Input RGB shape: {x_rgb.shape}")
    print(f"Input IR shape: {x_ir.shape}")
    print(f"C2f Homo feature output RGB shape: {out_homo[0].shape}")
    print(f"C2f Homo feature output IR shape: {out_homo[1].shape}")
    assert out_homo[0].shape == (batch_size, c_out, h, w), "C2f Homo feature RGB output dimension error"
    assert out_homo[1].shape == (batch_size, c_out, h, w), "C2f Homo feature IR output dimension error"
    print("PASS: C2f Homo feature extraction test")

    # Test heterogeneous feature C2f
    c2f_het = C2f_adalora_sym_m_het(c_in, c_out, n=3, shortcut=True, r=9, lora_alpha=9, lora_dropout=0.)
    out_het = c2f_het(x)

    print(f"C2f Het feature output RGB shape: {out_het[0].shape}")
    print(f"C2f Het feature output IR shape: {out_het[1].shape}")
    assert out_het[0].shape == (batch_size, c_out, h, w), "C2f Het feature RGB output dimension error"
    assert out_het[1].shape == (batch_size, c_out, h, w), "C2f Het feature IR output dimension error"
    print("PASS: C2f Het feature extraction test")


def test_add_rgb_ir():
    """Test Add_RGB and Add_IR"""
    print("\n" + "="*60)
    print("Test 3: Add_RGB and Add_IR")
    print("="*60)

    batch_size = 2
    c = 256
    h, w = 16, 16

    # Create input
    fpn_rgb = torch.randn(batch_size, c, h, w)
    fpn_ir = torch.randn(batch_size, c, h, w)
    het_rgb = torch.randn(batch_size, c, h, w)
    het_ir = torch.randn(batch_size, c, h, w)
    het_tuple = (het_rgb, het_ir)

    # Test Add_RGB
    add_rgb = Add_RGB(c)
    out_rgb = add_rgb([fpn_rgb, het_tuple])

    print(f"FPN RGB shape: {fpn_rgb.shape}")
    print(f"Het tuple RGB shape: {het_rgb.shape}")
    print(f"Add_RGB output shape: {out_rgb.shape}")
    assert out_rgb.shape == (batch_size, c, h, w), "Add_RGB output dimension error"
    print("PASS: Add_RGB test")

    # Test Add_IR
    add_ir = Add_IR(c)
    out_ir = add_ir([fpn_ir, het_tuple])

    print(f"FPN IR shape: {fpn_ir.shape}")
    print(f"Het tuple IR shape: {het_ir.shape}")
    print(f"Add_IR output shape: {out_ir.shape}")
    assert out_ir.shape == (batch_size, c, h, w), "Add_IR output dimension error"
    print("PASS: Add_IR test")


def test_concat_rgb_ir():
    """Test Concat_RGB and Concat_IR"""
    print("\n" + "="*60)
    print("Test 4: Concat_RGB and Concat_IR")
    print("="*60)

    batch_size = 2
    c = 256
    h, w = 16, 16

    # Create input
    feat1 = torch.randn(batch_size, c, h, w)
    feat2_rgb = torch.randn(batch_size, c, h, w)
    feat2_ir = torch.randn(batch_size, c, h, w)
    feat2_tuple = (feat2_rgb, feat2_ir)
    feat3_rgb = torch.randn(batch_size, c, h, w)
    feat3_ir = torch.randn(batch_size, c, h, w)
    feat3_tuple = (feat3_rgb, feat3_ir)

    # Test Concat_RGB
    concat_rgb = Concat_RGB(dimension=1)
    out_rgb = concat_rgb([feat1, feat2_tuple, feat3_tuple])

    print(f"Feat1 shape: {feat1.shape}")
    print(f"Feat2 tuple RGB shape: {feat2_rgb.shape}")
    print(f"Feat3 tuple RGB shape: {feat3_rgb.shape}")
    print(f"Concat_RGB output shape: {out_rgb.shape}")
    assert out_rgb.shape == (batch_size, c * 3, h, w), "Concat_RGB output dimension error"
    print("PASS: Concat_RGB test")

    # Test Concat_IR
    concat_ir = Concat_IR(dimension=1)
    out_ir = concat_ir([feat1, feat2_tuple, feat3_tuple])

    print(f"Concat_IR output shape: {out_ir.shape}")
    assert out_ir.shape == (batch_size, c * 3, h, w), "Concat_IR output dimension error"
    print("PASS: Concat_IR test")


def test_dcn_modules():
    """Test DCN related modules"""
    print("\n" + "="*60)
    print("Test 5: DCN Related Modules")
    print("="*60)

    batch_size = 2
    c = 512
    h, w = 16, 16

    # Create input
    feat_rgb = torch.randn(batch_size, c, h, w)
    feat_ir = torch.randn(batch_size, c, h, w)
    feat_tuple = (feat_rgb, feat_ir)

    # Test DCN_Offset_Conv
    dcn_offset = DCN_Offset_Conv(c, kernel_size=3)
    offset = dcn_offset(feat_tuple)

    print(f"Input RGB shape: {feat_rgb.shape}")
    print(f"Input IR shape: {feat_ir.shape}")
    print(f"DCN Offset output shape: {offset.shape}")
    assert offset.shape[0] == batch_size, "DCN Offset batch dimension error"
    assert offset.shape[1] == 2 * 3 * 3, "DCN Offset channel count error"
    assert offset.shape[2:] == (h, w), "DCN Offset spatial dimension error"
    print("PASS: DCN_Offset_Conv test")

    # Test DCN_Warp_RGB
    dcn_warp_rgb = DCN_Warp_RGB(c, kernel_size=3)
    warped_rgb = dcn_warp_rgb([feat_tuple, offset])

    print(f"DCN Warp RGB output shape: {warped_rgb.shape}")
    assert warped_rgb.shape == (batch_size, c, h, w), "DCN Warp RGB output dimension error"
    print("PASS: DCN_Warp_RGB test")

    # Test DCN_Warp_IR
    dcn_warp_ir = DCN_Warp_IR(c, kernel_size=3)
    warped_ir = dcn_warp_ir([feat_tuple, offset])

    print(f"DCN Warp IR output shape: {warped_ir.shape}")
    assert warped_ir.shape == (batch_size, c, h, w), "DCN Warp IR output dimension error"
    print("PASS: DCN_Warp_IR test")


def test_full_pipeline():
    """Test the full feature flow pipeline"""
    print("\n" + "="*60)
    print("Test 6: Full Feature Flow Pipeline")
    print("="*60)

    batch_size = 2
    h, w = 64, 64

    # Simulate Backbone output
    print("\n--- Backbone Stage ---")
    x_rgb = torch.randn(batch_size, 3, h, w)
    x_ir = torch.randn(batch_size, 3, h, w)
    x = (x_rgb, x_ir)

    # P3 level
    conv_homo_p3 = Conv_adalora_sym_m_homo(3, 256, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)
    conv_het_p3 = Conv_adalora_sym_m_het(3, 256, k=3, s=1, r=9, lora_alpha=9, lora_dropout=0.)

    homo_p3 = conv_homo_p3(x)
    het_p3 = conv_het_p3(x)

    print(f"P3 Homo feature RGB shape: {homo_p3[0].shape}")
    print(f"P3 Homo feature IR shape: {homo_p3[1].shape}")
    print(f"P3 Het feature RGB shape: {het_p3[0].shape}")
    print(f"P3 Het feature IR shape: {het_p3[1].shape}")

    # Simulate FPN output
    print("\n--- FPN Stage ---")
    fpn_p3_rgb = torch.randn(batch_size, 256, h, w)
    fpn_p3_ir = torch.randn(batch_size, 256, h, w)

    # PAN stage: heterogeneous feature injection
    print("\n--- PAN Stage: Het Feature Injection ---")
    add_rgb = Add_RGB(256)
    add_ir = Add_IR(256)

    pan_p3_rgb = add_rgb([fpn_p3_rgb, het_p3])
    pan_p3_ir = add_ir([fpn_p3_ir, het_p3])

    print(f"PAN P3 RGB shape: {pan_p3_rgb.shape}")
    print(f"PAN P3 IR shape: {pan_p3_ir.shape}")

    assert pan_p3_rgb.shape == (batch_size, 256, h, w), "PAN P3 RGB dimension error"
    assert pan_p3_ir.shape == (batch_size, 256, h, w), "PAN P3 IR dimension error"

    print("\nPASS: Full pipeline test")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Homo/Het Feature Extraction Modules")
    print("="*60)

    try:
        test_homo_het_conv()
        test_homo_het_c2f()
        test_add_rgb_ir()
        test_concat_rgb_ir()
        test_dcn_modules()
        test_full_pipeline()

        print("\n" + "="*60)
        print("PASS: All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\nFAIL: Test failed: {e}")
        import traceback
        traceback.print_exc()
