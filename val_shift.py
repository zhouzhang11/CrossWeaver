#!/usr/bin/env python3
"""
Validation script with shift support for robustness evaluation.
This script allows testing model robustness to spatial misalignment between IR and RGB images.
"""

import argparse
from ultralytics import YOLO_m

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO validation with spatial shift support')
    parser.add_argument('--weights', type=str, 
                        default='/home/sysspace/yanght/test/new/DroneVehicle/yolov8s_new_model10/weights/best.pt',
                        help='Path to model weights')
    parser.add_argument('--data', type=str, 
                        default='drone_vehicle_m.yaml',
                        help='Path to data yaml file')
    parser.add_argument('--shift_x', type=int, default=0,
                        help='X-axis shift in pixels (positive = right, negative = left)')
    parser.add_argument('--shift_y', type=int, default=0,
                        help='Y-axis shift in pixels (positive = down, negative = up)')
    parser.add_argument('--imgsz', type=int, default=800,
                        help='Image size for validation')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--project', type=str, default='DroneVehicle',
                        help='Project directory to save results')
    parser.add_argument('--name', type=str, default='shift_experiment',
                        help='Experiment name')
    parser.add_argument('--conf', type=float, default=None,
                        help='Confidence threshold (default: use model default)')
    parser.add_argument('--iou', type=float, default=None,
                        help='IoU threshold (default: use model default)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = YOLO_m(args.weights)
    
    # Prepare validation arguments
    val_args = {
        'data': args.data,
        'project': args.project,
        'name': f'{args.name}_x{args.shift_x:+d}_y{args.shift_y:+d}',  # e.g., shift_experiment_x+5_y-10
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'shift_x': args.shift_x,
        'shift_y': args.shift_y,
    }
    
    # Add optional parameters if specified
    if args.conf is not None:
        val_args['conf'] = args.conf
    if args.iou is not None:
        val_args['iou'] = args.iou
    
    # Print configuration
    print("\n" + "="*60)
    print("Validation Configuration:")
    print("="*60)
    print(f"Weights:    {args.weights}")
    print(f"Data:       {args.data}")
    print(f"Shift:      x={args.shift_x:+d}, y={args.shift_y:+d} pixels")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device:     GPU {args.device}")
    print(f"Save to:    {args.project}/{val_args['name']}")
    print("="*60 + "\n")
    
    # Run validation
    results = model.val(**val_args)
    
    # Print results summary
    print("\n" + "="*60)
    print("Validation Results:")
    print("="*60)
    print(f"Shift: x={args.shift_x:+d}, y={args.shift_y:+d}")
    if hasattr(results, 'box'):
        print(f"mAP50:     {results.box.map50:.4f}")
        print(f"mAP50-95:  {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall:    {results.box.mr:.4f}")
    print("="*60 + "\n")
    
    return results

if __name__ == '__main__':
    main()
