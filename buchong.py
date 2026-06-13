import os
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import shutil

# --- 1. 配置您的路径 ---
# dir_A: 您的新目录 (来源)
DIR_A = '/home/data1/yanght/DroneVehicle/home/data1/yanght/DroneVehicle/DroneVehicle_mm/val/ir/labels/'
# dir_B: 您用于比较的参考目录
DIR_B = '/home/data1/yanght/DroneVehicle/DroneVehicle_mm/val/ir/labels/'

# --- 2. (可选) 配置日志文件 ---
LOG_DETAILS_FILE = 'log_sync_details.txt'
LOG_SUMMARY_FILE = 'log_sync_summary.txt'

# --- 3. (可选) 配置比较容差 ---
TOLERANCE = 1e-6

def parse_yolo_obb_line(line_str):
    """
    解析 YOLO OBB 文本行。
    返回一个字典 {'class': int, 'coords': np.array, 'line': str} 或 None。
    """
    try:
        parts = line_str.strip().split()
        if len(parts) != 9:
            return None # 格式不正确
        class_id = int(parts[0])
        coords = np.array([float(p) for p in parts[1:]])
        return {
            'class': class_id,
            'coords': coords,
            'line': line_str.strip()
        }
    except Exception:
        return None

def compare_files_numerically(file_a_path, file_b_path):
    """
    使用数字容差 (np.isclose) 智能比较两个标签文件。
    返回: (identical_pairs, only_in_a, only_in_b)
    """
    list_a = []
    list_b = []
    
    if os.path.exists(file_a_path):
        with open(file_a_path, 'r') as f:
            for line in f:
                parsed = parse_yolo_obb_line(line)
                if parsed: list_a.append(parsed)
                
    if os.path.exists(file_b_path):
         with open(file_b_path, 'r') as f:
            for line in f:
                parsed = parse_yolo_obb_line(line)
                if parsed: list_b.append(parsed)

    identical_pairs = []
    only_in_a = []
    only_in_b = []
    
    matched_b_indices = set()

    for box_a in list_a:
        found_match = False
        for idx_b, box_b in enumerate(list_b):
            if idx_b in matched_b_indices:
                continue
            if (box_a['class'] == box_b['class'] and \
                np.all(np.isclose(box_a['coords'], box_b['coords'], atol=TOLERANCE))):
                identical_pairs.append((box_a['line'], box_b['line']))
                matched_b_indices.add(idx_b)
                found_match = True
                break
        if not found_match:
            only_in_a.append(box_a['line'])

    for idx_b, box_b in enumerate(list_b):
        if idx_b not in matched_b_indices:
            only_in_b.append(box_b['line'])
            
    return identical_pairs, only_in_a, only_in_b


def main_sync_with_confirm():
    """
    主函数：
    1. 分析 A 和 B 的差异并制定计划。
    2. 向用户显示计划并请求确认。
    3. (如果确认) 执行计划，将 A 的内容同步到 B。
    """
    print(f"--- 开始分析同步计划 A -> B ---")
    print(f"!! 警告：此脚本准备修改目录 B。将在执行前请求确认 !!")
    print(f"目录 A (来源): {DIR_A}")
    print(f"目录 B (目标): {DIR_B}")
    
    # --- 1. 路径和日志设置 ---
    if not os.path.isdir(DIR_A):
        print(f"错误: 目录 A 不存在: {DIR_A}", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(DIR_B):
        print(f"警告: 目录 B 不存在。正在创建: {DIR_B}")
        try:
            os.makedirs(DIR_B)
        except Exception as e:
            print(f"致命错误: 无法创建目录 B: {e}", file=sys.stderr)
            return

    with open(LOG_DETAILS_FILE, 'w') as f:
        f.write(f"# --- 详细分析日志 ---\n# 目录 A: {DIR_A}\n# 目录 B: {DIR_B}\n")
        f.write(f"# 容差 (Tolerance): {TOLERANCE}\n\n")

    # --- 阶段 1: 分析 (只读) ---
    print(f"\n--- 阶段 1: 正在分析差异 (只读模式) ---")
    
    try:
        files_to_check = sorted([f for f in os.listdir(DIR_A) if f.endswith('.txt')])
    except FileNotFoundError:
        print(f"错误: 目录 A 未找到: {DIR_A}", file=sys.stderr)
        return
        
    if not files_to_check:
        print(f"在 {DIR_A} 中未找到 .txt 文件。")
        return

    # 用于存储操作计划
    actions_to_perform = []
    
    # 统计数据
    total_identical_lines = 0
    total_only_in_b = 0
    total_lines_to_add = 0
    files_to_create = 0
    files_to_append = 0
    
    try:
        with open(LOG_DETAILS_FILE, 'a') as f_details:
            for filename in tqdm(files_to_check, desc="分析文件"):
                path_a = os.path.join(DIR_A, filename)
                path_b = os.path.join(DIR_B, filename)
                
                f_details.write(f"\n=====================\n")
                f_details.write(f"FILE: {filename}\n")
                
                # --- 1.1: B 中文件不存在 ---
                if not os.path.exists(path_b):
                    lines_in_a = []
                    with open(path_a, 'r') as f:
                         lines_in_a = [line.strip() for line in f if line.strip()]
                    
                    f_details.write(f"  > 发现: 在目录 B 中缺失。\n")
                    f_details.write(f"  > 计划: [创建] 文件 {filename}，包含 {len(lines_in_a)} 行。\n")
                    
                    # 添加到操作计划
                    actions_to_perform.append({
                        'type': 'CREATE',
                        'filename': filename,
                        'path_a': path_a,
                        'path_b': path_b,
                        'line_count': len(lines_in_a)
                    })
                    
                    files_to_create += 1
                    total_lines_to_add += len(lines_in_a)
                    continue
                
                # --- 1.2: B 中文件已存在，执行比较 ---
                identical, only_a, only_b = compare_files_numerically(path_a, path_b)
                
                # 累加统计
                total_identical_lines += len(identical)
                total_only_in_b += len(only_b)
                
                if not only_a and not only_b:
                    f_details.write(f"  > 发现: 完美匹配 (共 {len(identical)} 行)。\n")
                    f_details.write(f"  > 计划: 无操作。\n")
                else:
                    f_details.write(f"  > 发现: 发现差异。\n")
                    for line in only_b:
                        f_details.write(f"    > (仅在 B): {line}\n")
                    
                    if only_a:
                        for line in only_a:
                            f_details.write(f"    > (仅在 A): {line}\n")
                        
                        f_details.write(f"  > 计划: [追加] {len(only_a)} 行到 {filename}。\n")
                        
                        # 添加到操作计划
                        actions_to_perform.append({
                            'type': 'APPEND',
                            'filename': filename,
                            'path_b': path_b,
                            'lines_to_add': only_a,
                            'line_count': len(only_a)
                        })
                        
                        files_to_append += 1
                        total_lines_to_add += len(only_a)
                    else:
                        f_details.write(f"  > 计划: 无操作 (仅 B 有多余行)。\n")

    except Exception as e:
        print(f"\n分析时发生错误: {e}")
        return

    # --- 阶段 2: 预览和确认 ---
    print(f"\n--- 阶段 2: 分析完成。预览即将执行的操作 ---")
    
    if not actions_to_perform:
        print("分析完毕：目录 A 和 B 已经同步。无需执行任何操作。")
        # (仍然继续执行以写入总结日志)
    else:
        print("以下操作将被执行：\n")
        for action in actions_to_perform:
            if action['type'] == 'CREATE':
                print(f"  [创建] 将 {action['path_a']} \n    -> 复制到 {action['path_b']} (共 {action['line_count']} 行)")
            elif action['type'] == 'APPEND':
                print(f"  [追加] 将 {action['line_count']} 行新标注添加到 \n    -> {action['path_b']}")
        
        print("\n" + "=" * 50)
        print("--- 预览总结 ---")
        print(f"将创建 {files_to_create} 个新文件。")
        print(f"将追加 {files_to_append} 个现有文件。")
        print(f"总共将从 A 添加 {total_lines_to_add} 行标注到 B。")
        print("=" * 50)
        print(f"\n详细分析日志已保存到: {os.path.abspath(LOG_DETAILS_FILE)}")

        # --- 请求确认 ---
        confirmation = input("\n是否执行以上所有操作？请输入 'yes' 以继续: ").strip().lower()
    
        if confirmation != 'yes':
            print("操作已取消。未对目录 B 进行任何修改。")
            write_summary_log(files_to_check, files_to_create, files_to_append, total_identical_lines, total_lines_to_add, total_only_in_b, executed=False)
            return
    
    # --- 阶段 3: 执行 (仅在确认后) ---
    if actions_to_perform:
        print(f"\n--- 阶段 3: 正在执行同步操作 ---")
        try:
            for action in tqdm(actions_to_perform, desc="执行同步"):
                if action['type'] == 'CREATE':
                    shutil.copyfile(action['path_a'], action['path_b'])
                
                elif action['type'] == 'APPEND':
                    with open(action['path_b'], 'a') as f_b_append:
                        for line_to_add in action['lines_to_add']:
                            f_b_append.write(line_to_add + '\n')
            
            print("\n--- 同步执行完毕 ---")
            
        except Exception as e:
            print(f"\n执行操作时发生致命错误: {e}", file=sys.stderr)
            print("部分文件可能已修改！请检查日志。")
            write_summary_log(files_to_check, files_to_create, files_to_append, total_identical_lines, total_lines_to_add, total_only_in_b, executed=False, error=str(e))
            return

    # --- 阶段 4: 写入最终总结 ---
    write_summary_log(files_to_check, files_to_create, files_to_append, total_identical_lines, total_lines_to_add, total_only_in_b, executed=bool(actions_to_perform))


def write_summary_log(files_to_check, files_created, files_appended, total_identical, total_added, total_only_b, executed, error=None):
    """
    将最终的总结写入控制台和日志文件。
    """
    status_msg = "操作已成功执行" if executed else "操作被取消或无需操作"
    if error:
        status_msg = f"操作因错误而失败: {error}"
    
    print("\n" + "=" * 50)
    print("--- 最终总结报告 ---")
    print(f"处理状态: {status_msg}")
    print(f"总共分析的文件数 (基于 A): {len(files_to_check)}")
    print("-" * 50)
    print(f"原已相同的标注行: {total_identical}")
    print(f"仅存在于 B (未修改) 的标注行: {total_only_b}")
    print(f"--- 计划/执行的操作 ---")
    print(f"在 B 中新创建的文件: {files_created if executed else f'{files_created} (未执行)'}")
    print(f"在 B 中被追加内容的文件: {files_appended if executed else f'{files_appended} (未执行)'}")
    print(f"从 A 新增到 B 的标注行 (总计): {total_added if executed else f'{total_added} (未执行)'}")
    print("=" * 50)
    
    try:
        with open(LOG_SUMMARY_FILE, 'w') as f: # 'w' 模式覆盖旧的
            f.write(f"# --- 总结同步日志 ---\n")
            f.write(f"处理状态: {status_msg}\n\n")
            f.write(f"目录 A (来源): {DIR_A}\n")
            f.write(f"目录 B (目标): {DIR_B}\n")
            f.write(f"容差: {TOLERANCE}\n\n")
            f.write(f"总共分析的文件数 (基于 A): {len(files_to_check)}\n")
            f.write("-" * 50 + "\n")
            f.write(f"原已相同的标注行: {total_identical}\n")
            f.write(f"仅存在于 B (未修改) 的标注行: {total_only_b}\n")
            f.write(f"--- 计划/执行的操作 ---\n")
            f.write(f"在 B 中新创建的文件: {files_created if executed else f'{files_created} (未执行)'}\n")
            f.write(f"在 B 中被追加内容的文件: {files_appended if executed else f'{files_appended} (未执行)'}\n")
            f.write(f"从 A 新增到 B 的标注行 (总计): {total_added if executed else f'{total_added} (未执行)'}\n")
            
        print(f"\n详细分析日志已保存到: {os.path.abspath(LOG_DETAILS_FILE)}")
        print(f"总结日志已保存到: {os.path.abspath(LOG_SUMMARY_FILE)}")
        
    except Exception as e:
        print(f"\n错误: 无法写入总结日志文件: {e}", file=sys.stderr)

if __name__ == "__main__":
    main_sync_with_confirm()