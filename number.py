import os
import glob
from collections import defaultdict, Counter
from PIL import Image  # 用于读取图片尺寸

# --- 关键路径设置：请为你新的 OBB 数据集更新这些路径 ---
# --- 关键路径设置 ---
# 1. 你的标签文件夹路径
LABEL_DIR = "/home/data1/yanght/DroneVehicle/DroneVehicle_mm/train/rgb/labels/"
# 2. 你的图片文件夹路径 (新添加)
IMAGE_DIR = "/home/data1/yanght/DroneVehicle/DroneVehicle_mm/train/rgb/images/"
# ---------------------------------------------------------

# COCO 官方的面积标准 (像素)
COCO_SMALL_AREA = 32 * 32   # 1024
COCO_MEDIUM_AREA = 96 * 96  # 9216

def get_pixel_area(normalized_w, normalized_h, img_w, img_h):
    """将归一化的YOLO尺寸转换为像素面积"""
    pixel_w = normalized_w * img_w
    pixel_h = normalized_h * img_h
    return pixel_w * pixel_h

def categorize_area(area):
    """根据COCO标准分类"""
    if area < COCO_SMALL_AREA:
        return 'small'
    elif area < COCO_MEDIUM_AREA:
        return 'medium'
    else:
        return 'large'

def find_corresponding_image(label_file_path, image_dir):
    """在 image_dir 中查找与 label_file 匹配的图片文件"""
    base_name = os.path.splitext(os.path.basename(label_file_path))[0]
    possible_images = glob.glob(os.path.join(image_dir, base_name + '.*'))
    for img_path in possible_images:
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            return img_path
    return None

def get_image_dimensions(image_path):
    """使用 Pillow 快速读取图片的 (width, height)"""
    try:
        with Image.open(image_path) as img:
            return img.size  # 返回 (width, height) 元组
    except Exception as e:
        print(f"  > 警告: 无法读取图像 {os.path.basename(image_path)}. 错误: {e}")
        return None

def analyze_dataset(label_dir, image_dir):
    """主分析函数"""
    
    # 检查路径是否存在
    if not os.path.isdir(label_dir):
        print(f"错误：找不到标签目录 {label_dir}")
        print("‼️ 请确保你已在脚本中更新了 LABEL_DIR 变量")
        return
    if not os.path.isdir(image_dir):
        print(f"错误：找不到图片目录 {image_dir}")
        print("‼️ 请确保你已在脚本中更新了 IMAGE_DIR 变量")
        return

    # 1. 找到所有 label 文件
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    if not label_files:
        print(f"错误：在 {label_dir} 中没有找到任何 .txt 标签文件。")
        return

    print(f"--- OBB 数据集分析 ---")
    print(f"标签目录: {label_dir}")
    print(f"图片目录: {image_dir}")
    print(f"分析文件数: {len(label_files)} 个")
    print(f"标准: Small < {COCO_SMALL_AREA}px, Medium < {COCO_MEDIUM_AREA}px\n")

    # 2. 初始化计数器
    total_counts = Counter()
    class_counts = defaultdict(Counter)
    total_instances = 0
    files_without_images = 0
    
    print("正在处理标签文件并读取图片尺寸 (这可能需要一两分钟)...")

    # 3. 遍历并处理每个文件
    for i, file_path in enumerate(label_files):
        
        if (i + 1) % 200 == 0:
            print(f"  ...已处理 {i+1}/{len(label_files)} 个文件")

        image_path = find_corresponding_image(file_path, image_dir)
        if not image_path:
            files_without_images += 1
            continue
        
        dimensions = get_image_dimensions(image_path)
        if not dimensions:
            continue
        
        img_w, img_h = dimensions

        # 3.3. 读取标签文件并计算
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    parts = line.split()
                    
                    # ‼️ 核心修改点：检查9个值 (class_id + 4*x + 4*y)
                    if len(parts) != 9:
                        print(f"警告: 跳过格式错误的行 (应为9个值): {line} in {file_path}")
                        continue

                    class_id = int(parts[0])
                    
                    # 提取所有归一化的 x 和 y 坐标
                    try:
                        norm_x_coords = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
                        norm_y_coords = [float(parts[2]), float(parts[4]), float(parts[6]), float(parts[8])]
                    except ValueError:
                        print(f"警告: 跳过坐标无法解析的行: {line} in {file_path}")
                        continue
                    
                    # 找到外接水平框 (AABB)
                    norm_x_min = min(norm_x_coords)
                    norm_x_max = max(norm_x_coords)
                    norm_y_min = min(norm_y_coords)
                    norm_y_max = max(norm_y_coords)
                    
                    # 计算 AABB 的归一化宽度和高度
                    norm_w = norm_x_max - norm_x_min
                    norm_h = norm_y_max - norm_y_min

                    # 核心计算：使用真实的 img_w 和 img_h
                    area = get_pixel_area(norm_w, norm_h, img_w, img_h)
                    category = categorize_area(area)

                    total_instances += 1
                    total_counts[category] += 1
                    class_counts[class_id][category] += 1

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 4. 打印结果
    print(f"\n--- 统计完成 ---")
    
    if files_without_images > 0:
        print(f"⚠️ 警告: 有 {files_without_images} 个标签文件在图片目录中找不到对应的图片。")

    if total_instances == 0:
        print("没有找到任何检测框。")
        return

    print(f"--- 总体目标分布 (共 {total_instances} 个) ---")
    
    total_small = total_counts['small']
    total_medium = total_counts['medium']
    total_large = total_counts['large']
    
    print(f"  小目标 (Small):  {total_small:8d} ({total_small/total_instances:.2%})")
    print(f"  中目标 (Medium): {total_medium:8d} ({total_medium/total_instances:.2%})")
    print(f"  大目标 (Large):  {total_large:8d} ({total_large/total_instances:.2%})")

    print("\n--- 按类别的详细清单 ---")
    
    sorted_class_ids = sorted(class_counts.keys())
    
    for class_id in sorted_class_ids:
        counts = class_counts[class_id]
        class_total = counts['small'] + counts['medium'] + counts['large']
        
        if class_total == 0: continue
        
        print(f"\n[类别 {class_id}] (共 {class_total} 个)")
        print(f"  - Small:  {counts['small']:8d} ({counts['small']/class_total:.2%})")
        print(f"  - Medium: {counts['medium']:8d} ({counts['medium']/class_total:.2%})")
        print(f"  - Large:  {counts['large']:8d} ({counts['large']/class_total:.2%})")

if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        print("错误: 找不到 'Pillow' (PIL) 库。")
        print("请运行 'pip install Pillow' 来安装它。")
        exit(1)
        
    analyze_dataset(LABEL_DIR, IMAGE_DIR)