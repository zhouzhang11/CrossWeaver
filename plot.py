#!/usr/bin/env python3
"""
Plot ground truth labels on validation images
将验证集的ground truth标签画到图像上并输出
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import ops


def load_dataset_config(data_yaml):
    """加载数据集配置文件"""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data


def read_obb_label(label_path):
    """
    读取OBB格式的标签文件
    格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
    返回: list of [class_id, [x1,y1,x2,y2,x3,y3,x4,y4]]
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
    将四个角点坐标转换为中心点+宽高+角度格式
    coords: [x1,y1,x2,y2,x3,y3,x4,y4]
    返回: [cx, cy, w, h, angle]
    """
    # 转换为numpy数组并reshape
    points = np.array(coords).reshape(4, 2)
    
    # 计算中心点
    cx = points[:, 0].mean()
    cy = points[:, 1].mean()
    
    # 计算宽度和高度（使用边长）
    w = np.linalg.norm(points[1] - points[0])
    h = np.linalg.norm(points[2] - points[1])
    
    # 计算角度（使用第一条边的角度）
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle = np.arctan2(dy, dx)
    
    return [cx, cy, w, h, angle]


def denormalize_coords(coords, img_width, img_height):
    """
    将归一化坐标转换为像素坐标
    coords: [x1,y1,x2,y2,x3,y3,x4,y4] (normalized 0-1)
    返回: [x1,y1,x2,y2,x3,y3,x4,y4] (pixels)
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
    绘制ground truth标签到图像上
    
    Args:
        data_yaml: 数据集配置文件路径（可选，如果提供rgb_dir和ir_dir则不需要）
        output_dir: 输出目录
        modality: 'rgb', 'ir', 或 'both'
        rgb_dir: RGB图像目录路径（可选，直接指定）
        ir_dir: IR图像目录路径（可选，直接指定）
        names_dict: 类别名称字典（可选，如果不提供则从data_yaml读取）
    """
    # 如果提供了直接路径，使用直接路径
    if rgb_dir or ir_dir:
        val_rgb_path = Path(rgb_dir) if rgb_dir else None
        val_ir_path = Path(ir_dir) if ir_dir else None
        
        # 类别名称
        if names_dict:
            names = names_dict
        elif data_yaml:
            data = load_dataset_config(data_yaml)
            names = data.get('names', {})
        else:
            # 默认类别名称
            names = {i: f'class_{i}' for i in range(10)}
    else:
        # 从配置文件加载
        if not data_yaml:
            raise ValueError("必须提供 data_yaml 或 rgb_dir/ir_dir")
        
        data = load_dataset_config(data_yaml)
        names = data.get('names', {})
        val_rgb_path = Path(data.get('val_rgb', ''))
        val_ir_path = Path(data.get('val_ir', ''))
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_rgb_dir = output_path / 'rgb'
    output_ir_dir = output_path / 'ir'
    
    if modality in ['rgb', 'both']:
        output_rgb_dir.mkdir(parents=True, exist_ok=True)
    if modality in ['ir', 'both']:
        output_ir_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_path}")
    print(f"类别数量: {len(names)}")
    print(f"类别: {names}")
    
    # 获取图像和标签路径
    if modality in ['rgb', 'both']:
        # RGB图像
        rgb_images_dir = val_rgb_path / 'images'
        rgb_labels_dir = val_rgb_path / 'labels'
        
        if rgb_images_dir.exists():
            rgb_image_files = sorted(list(rgb_images_dir.glob('*.jpg')) + 
                                    list(rgb_images_dir.glob('*.png')))
            print(f"\n处理RGB图像: {len(rgb_image_files)} 张")
            
            for img_path in tqdm(rgb_image_files, desc="RGB"):
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # 读取对应的标签
                label_path = rgb_labels_dir / (img_path.stem + '.txt')
                labels = read_obb_label(label_path)
                
                # 创建标注器
                annotator = Annotator(img, line_width=2)
                
                # 绘制每个标签
                for class_id, coords in labels:
                    # 反归一化坐标
                    pixel_coords = denormalize_coords(coords, w, h)
                    
                    # 转换为四个角点列表
                    corners = [[pixel_coords[i], pixel_coords[i+1]] 
                              for i in range(0, 8, 2)]
                    
                    # 获取类别信息
                    class_name = names.get(class_id, str(class_id))
                    color = colors(class_id, True)
                    
                    # 绘制旋转框（只显示类别名称）
                    label = f"{class_name}"
                    annotator.box_label(corners, label, color=color, rotated=True)
                
                # 保存结果
                result = annotator.result()
                output_file = output_rgb_dir / img_path.name
                cv2.imwrite(str(output_file), result)
    
    if modality in ['ir', 'both']:
        # IR图像
        ir_images_dir = val_ir_path / 'images'
        ir_labels_dir = val_ir_path / 'labels'
        
        if ir_images_dir.exists():
            ir_image_files = sorted(list(ir_images_dir.glob('*.jpg')) + 
                                   list(ir_images_dir.glob('*.png')))
            print(f"\n处理IR图像: {len(ir_image_files)} 张")
            
            for img_path in tqdm(ir_image_files, desc="IR"):
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # 读取对应的标签
                label_path = ir_labels_dir / (img_path.stem + '.txt')
                labels = read_obb_label(label_path)
                
                # 创建标注器
                annotator = Annotator(img, line_width=2)
                
                # 绘制每个标签
                for class_id, coords in labels:
                    # 反归一化坐标
                    pixel_coords = denormalize_coords(coords, w, h)
                    
                    # 转换为四个角点列表
                    corners = [[pixel_coords[i], pixel_coords[i+1]] 
                              for i in range(0, 8, 2)]
                    
                    # 获取类别信息
                    class_name = names.get(class_id, str(class_id))
                    color = colors(class_id, True)
                    
                    # 绘制旋转框（只显示类别名称）
                    label = f"{class_name}"
                    annotator.box_label(corners, label, color=color, rotated=True)
                
                # 保存结果
                result = annotator.result()
                output_file = output_ir_dir / img_path.name
                cv2.imwrite(str(output_file), result)
    
    print(f"\n✓ 完成！结果已保存到: {output_path}")
    if modality in ['rgb', 'both']:
        print(f"  RGB: {output_rgb_dir}")
    if modality in ['ir', 'both']:
        print(f"  IR: {output_ir_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot ground truth labels on validation images')
    parser.add_argument('--data', type=str, default='/home/sysspace/yanght/test/new/ultralytics/cfg/datasets/drone_vehicle_m.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--output', type=str, default='ground_truth_vis',
                       help='输出目录')
    parser.add_argument('--modality', type=str, default='both',
                       choices=['rgb', 'ir', 'both'],
                       help='处理的模态: rgb, ir, 或 both')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Ground Truth 可视化工具")
    print("="*60)
    print(f"数据集配置: {args.data}")
    print(f"输出目录: {args.output}")
    print(f"处理模态: {args.modality}")
    print("="*60)
    
    plot_ground_truth(args.data, args.output, args.modality)
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


if __name__ == '__main__':
    main()
