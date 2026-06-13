#!/usr/bin/env python3
"""
Plot ground truth labels on validation images
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import ops


def load_dataset_config(data_yaml):
    """Load dataset configuration file"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data


def read_obb_label(label_path):
    """
    Read OBB format label file
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    Returns: list of [class_id, [x1,y1,x2,y2,x3,y3,x4,y4]]
    """
    labels = []
    if not Path(label_path).exists():
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:  # class_id + 8 coordinates
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:9]]
                labels.append([class_id, coords])
    return labels


def xyxyxyxy_to_xywhr(coords):
    """
    Convert four corner points to center + width/height + angle format
    coords: [x1,y1,x2,y2,x3,y3,x4,y4]
    Returns: [cx, cy, w, h, angle]
    """
    # Convert to numpy array and reshape
    points = np.array(coords).reshape(4, 2)

    # Compute center point
    cx = points[:, 0].mean()
    cy = points[:, 1].mean()

    # Compute width and height (using edge lengths)
    w = np.linalg.norm(points[1] - points[0])
    h = np.linalg.norm(points[2] - points[1])

    # Compute angle (using first edge angle)
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle = np.arctan2(dy, dx)

    return [cx, cy, w, h, angle]


def denormalize_coords(coords, img_width, img_height):
    """
    Convert normalized coordinates to pixel coordinates
    coords: [x1,y1,x2,y2,x3,y3,x4,y4] (normalized 0-1)
    Returns: [x1,y1,x2,y2,x3,y3,x4,y4] (pixels)
    """
    denorm_coords = []
    for i, val in enumerate(coords):
        if i % 2 == 0:  # x coordinate
            denorm_coords.append(val * img_width)
        else:  # y coordinate
            denorm_coords.append(val * img_height)
    return denorm_coords


def plot_ground_truth(data_yaml=None, output_dir='ground_truth_vis', modality='both',
                     rgb_dir=None, ir_dir=None, names_dict=None):
    """
    Plot ground truth labels onto images

    Args:
        data_yaml: Dataset configuration file path (optional, not needed if rgb_dir/ir_dir provided)
        output_dir: Output directory
        modality: 'rgb', 'ir', or 'both'
        rgb_dir: RGB image directory path (optional, directly specified)
        ir_dir: IR image directory path (optional, directly specified)
        names_dict: Class name dictionary (optional, read from data_yaml if not provided)
    """
    # Use direct paths if provided
    if rgb_dir or ir_dir:
        val_rgb_path = Path(rgb_dir) if rgb_dir else None
        val_ir_path = Path(ir_dir) if ir_dir else None

        # Class names
        if names_dict:
            names = names_dict
        elif data_yaml:
            data = load_dataset_config(data_yaml)
            names = data.get('names', {})
        else:
            # Default class names
            names = {i: f'class_{i}' for i in range(10)}
    else:
        # Load from config file
        if not data_yaml:
            raise ValueError("Must provide data_yaml or rgb_dir/ir_dir")

        data = load_dataset_config(data_yaml)
        names = data.get('names', {})
        val_rgb_path = Path(data.get('val_rgb', ''))
        val_ir_path = Path(data.get('val_ir', ''))

    # Create output directories
    output_path = Path(output_dir)
    output_rgb_dir = output_path / 'rgb'
    output_ir_dir = output_path / 'ir'

    if modality in ['rgb', 'both']:
        output_rgb_dir.mkdir(parents=True, exist_ok=True)
    if modality in ['ir', 'both']:
        output_ir_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_path}")
    print(f"Number of classes: {len(names)}")
    print(f"Classes: {names}")

    # Get image and label paths
    if modality in ['rgb', 'both']:
        # RGB images
        rgb_images_dir = val_rgb_path / 'images'
        rgb_labels_dir = val_rgb_path / 'labels'

        if rgb_images_dir.exists():
            rgb_image_files = sorted(list(rgb_images_dir.glob('*.jpg')) +
                                    list(rgb_images_dir.glob('*.png')))
            print(f"\nProcessing RGB images: {len(rgb_image_files)} images")

            for img_path in tqdm(rgb_image_files, desc="RGB"):
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Read corresponding label
                label_path = rgb_labels_dir / (img_path.stem + '.txt')
                labels = read_obb_label(label_path)

                # Create annotator
                annotator = Annotator(img, line_width=2)

                # Draw each label
                for class_id, coords in labels:
                    # Denormalize coordinates
                    pixel_coords = denormalize_coords(coords, w, h)

                    # Convert to four corner points list
                    corners = [[pixel_coords[i], pixel_coords[i+1]]
                              for i in range(0, 8, 2)]

                    # Get class info
                    class_name = names.get(class_id, str(class_id))
                    color = colors(class_id, True)

                    # Draw rotated box (show only class name)
                    label = f"{class_name}"
                    annotator.box_label(corners, label, color=color, rotated=True)

                # Save result
                result = annotator.result()
                output_file = output_rgb_dir / img_path.name
                cv2.imwrite(str(output_file), result)

    if modality in ['ir', 'both']:
        # IR images
        ir_images_dir = val_ir_path / 'images'
        ir_labels_dir = val_ir_path / 'labels'

        if ir_images_dir.exists():
            ir_image_files = sorted(list(ir_images_dir.glob('*.jpg')) +
                                   list(ir_images_dir.glob('*.png')))
            print(f"\nProcessing IR images: {len(ir_image_files)} images")

            for img_path in tqdm(ir_image_files, desc="IR"):
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Read corresponding label
                label_path = ir_labels_dir / (img_path.stem + '.txt')
                labels = read_obb_label(label_path)

                # Create annotator
                annotator = Annotator(img, line_width=2)

                # Draw each label
                for class_id, coords in labels:
                    # Denormalize coordinates
                    pixel_coords = denormalize_coords(coords, w, h)

                    # Convert to four corner points list
                    corners = [[pixel_coords[i], pixel_coords[i+1]]
                              for i in range(0, 8, 2)]

                    # Get class info
                    class_name = names.get(class_id, str(class_id))
                    color = colors(class_id, True)

                    # Draw rotated box (show only class name)
                    label = f"{class_name}"
                    annotator.box_label(corners, label, color=color, rotated=True)

                # Save result
                result = annotator.result()
                output_file = output_ir_dir / img_path.name
                cv2.imwrite(str(output_file), result)

    print(f"\nDone! Results saved to: {output_path}")
    if modality in ['rgb', 'both']:
        print(f"  RGB: {output_rgb_dir}")
    if modality in ['ir', 'both']:
        print(f"  IR: {output_ir_dir}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Plot ground truth labels on validation images')
    parser.add_argument('--data', type=str, default='/home/sysspace/yanght/test/new/ultralytics/cfg/datasets/drone_vehicle_m.yaml',
                       help='Dataset configuration file path')
    parser.add_argument('--output', type=str, default='ground_truth_vis',
                       help='Output directory')
    parser.add_argument('--modality', type=str, default='both',
                       choices=['rgb', 'ir', 'both'],
                       help='Modality to process: rgb, ir, or both')

    args = parser.parse_args()

    print("="*60)
    print("Ground Truth Visualization Tool")
    print("="*60)
    print(f"Dataset config: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Modality: {args.modality}")
    print("="*60)

    plot_ground_truth(args.data, args.output, args.modality)

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
