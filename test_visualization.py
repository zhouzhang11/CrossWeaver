#!/usr/bin/env python3
"""
Test script to verify the visualization functionality
"""
from ultralytics import YOLO_m
from pathlib import Path

# Load model
model_path = '/home/sysspace/yanght/test/new/Ablation/M3FD8/weights/best.pt'
print(f"Loading model from: {model_path}")
model = YOLO_m(model_path)

# Run validation with visualization
data = 'M3FD.yaml'
print(f"\nRunning validation with save_vis=True...")
print(f"Dataset: {data}")
print(f"Output will be saved to: DroneVehicle/val_yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8_best/out/")
print(f"  - RGB images: DroneVehicle/val_yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8_best/out/rgb/")
print(f"  - IR images: DroneVehicle/val_yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8_best/out/ir/")

results = model.val(
    data=data,
    project='DroneVehicle',
    name='val_yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8_best',
    imgsz=800,
    batch=16,
    device=0,
    save_vis=True,  # Enable saving visualized predictions
)

print("\n" + "="*60)
print("Validation completed!")
print("="*60)
print(f"\nResults:")
for key, value in results.items():
    print(f"  {key}: {value}")

# Check if output directories were created
out_dir = Path('DroneVehicle/val_yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8_best/out')
if out_dir.exists():
    rgb_dir = out_dir / 'rgb'
    ir_dir = out_dir / 'ir'
    
    if rgb_dir.exists():
        rgb_count = len(list(rgb_dir.glob('*.jpg'))) + len(list(rgb_dir.glob('*.png')))
        print(f"\n✓ RGB visualizations saved: {rgb_count} images in {rgb_dir}")
    else:
        print(f"\n✗ RGB directory not found: {rgb_dir}")
    
    if ir_dir.exists():
        ir_count = len(list(ir_dir.glob('*.jpg'))) + len(list(ir_dir.glob('*.png')))
        print(f"✓ IR visualizations saved: {ir_count} images in {ir_dir}")
    else:
        print(f"✗ IR directory not found: {ir_dir}")
else:
    print(f"\n✗ Output directory not found: {out_dir}")

print("\n" + "="*60)
print("Test completed!")
print("="*60)
