import json
import os
from collections import defaultdict
from tqdm import tqdm
import sys

# --- 1. 配置路径 ---
JSON_FILE = '/home/sysspace/yanght/test/DPDETR/DPDETR-main/dataset/rbox_Drone_Vehicle/val_ir_segmentation.json'
OUTPUT_DIR = '/home/sysspace/yanght/test/DPDETR/DPDETR-main/dataset/val/ir/label'
# --------------------

def convert_coco_obb_to_yolo():
    """
    将 COCO OBB JSON 格式转换为 YOLO OBB .txt 格式。
    - "segmentation" 字段被假定为 [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    
    print(f"--- COCO OBB 转换为 YOLO OBB ---")
    print(f"加载 JSON 文件: {JSON_FILE}...")
    
    # --- 2. 加载 JSON 文件 ---
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: JSON 文件未找到于 {JSON_FILE}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 无法解码 JSON。请检查文件内容是否正确。")
        sys.exit(1)

    # --- 3. 创建映射表 (用于高效查找) ---
    print("创建 image ID -> info (width, height, filename) 映射...")
    image_id_to_info = {}
    for image in data['images']:
        image_id_to_info[image['id']] = {
            'width': image['width'],
            'height': image['height'],
            'file_name': image['file_name']
        }
    
    print("按 image ID 分组标注...")
    annotations_by_image = defaultdict(list)
    
    # 过滤有效的 OBB 标注
    for ann in data['annotations']:
        seg = ann.get('segmentation')
        if not seg:
            continue

        seg_points = []
        # COCO 格式的 segmentation 可能是 [[x1,y1...]]
        if isinstance(seg, list) and len(seg) == 1 and isinstance(seg[0], list):
             seg_points = seg[0]
        # 也可能是您示例中的扁平格式 [x1,y1...]
        elif isinstance(seg, list) and len(seg) > 1:
             seg_points = seg
        else:
             continue # 跳过无效格式

        # 我们只处理 8-点 OBB 标注
        if len(seg_points) == 8:
            ann['segmentation_points'] = seg_points # 存储处理后的 8 个点
            annotations_by_image[ann['image_id']].append(ann)

    # --- 4. 创建输出目录 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录已确认: {OUTPUT_DIR}")

    # --- 5. 循环转换并写入文件 ---
    print(f"开始转换 {len(image_id_to_info)} 张图片的标注...")
    converted_count = 0
    skipped_count = 0

    # 使用 tqdm 创建进度条
    for image_id, info in tqdm(image_id_to_info.items()):
        img_w = info['width']
        img_h = info['height']
        file_name = info['file_name']
        
        # 健壮性检查，防止除以零
        if img_w == 0 or img_h == 0:
            print(f"警告: 图片 {file_name} (ID: {image_id}) 的宽度或高度为 0，已跳过。")
            skipped_count += 1
            continue

        # 获取此图片的所有标注
        annotations_list = annotations_by_image.get(image_id, [])
        
        yolo_lines = []
        
        for ann in annotations_list:
            category_id = ann['category_id']
            category_id = category_id - 1
            
            # 使用我们预处理好的 8 个点
            points = ann['segmentation_points'] 
            
            try:
                # [x1, y1, x2, y2, x3, y3, x4, y4]
                x1, y1, x2, y2, x3, y3, x4, y4 = points
                
                # --- 归一化 ---
                x1_norm = x1 / img_w
                y1_norm = y1 / img_h
                x2_norm = x2 / img_w
                y2_norm = y2 / img_h
                x3_norm = x3 / img_w
                y3_norm = y3 / img_h
                x4_norm = x4 / img_w
                y4_norm = y4 / img_h
                
                # 格式: class x1 y1 x2 y2 x3 y3 x4 y4 (保留8位小数)
                line = (f"{category_id} {x1_norm:.8f} {y1_norm:.8f} {x2_norm:.8f} {y2_norm:.8f} "
                        f"{x3_norm:.8f} {y3_norm:.8f} {x4_norm:.8f} {y4_norm:.8f}")
                yolo_lines.append(line)
                
            except Exception as e:
                print(f"错误: 处理标注 {ann['id']} (图片: {file_name}) 时出错: {e}")

        # --- 6. 写入 .txt 文件 ---
        # 从 "00001.jpg" 生成 "00001.txt"
        txt_name = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(OUTPUT_DIR, txt_name)
        
        try:
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n") # 如果文件非空，添加一个尾随换行符
            converted_count += 1
        except Exception as e:
            print(f"错误: 写入文件 {txt_path} 时出错: {e}")
            skipped_count += 1
            
    print("\n--- 转换完成 ---")
    print(f"成功转换并保存: {converted_count} 个文件。")
    print(f"跳过 (因错误或图像尺寸为0): {skipped_count} 个文件。")

if __name__ == "__main__":
    convert_coco_obb_to_yolo()