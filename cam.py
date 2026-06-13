"""
Shared-Backbone (AdaLoRA-Sym-M) Grad-CAM Visualization Script
(版本 8.4：修复 'save_gradient' 钩子逻辑 + 适配 'build_yolo_dataset_m')

用于为共享Backbone（如AdaLoRA-Sym-M）生成RGB和IR的类激活图(CAM)，
该模型接收 (x_rgb, x_ir) tuple 输入，并在共享层中输出 (feat_rgb, feat_ir) tuple。

将所有CAM叠加在IR原图上进行比较，并计算 P1-P5 上的 *全局* 模态相似度。
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    from ultralytics.data import build_dataloader, build_yolo_dataset_m, YOLODataset_m
except ImportError:
    print("错误：无法导入 'build_yolo_dataset_m' 或 'YOLODataset_m'。")
    print("请确保它们在 'ultralytics.data' 中定义，或将它们的定义粘贴到此脚本的顶部。")
    exit()

from ultralytics.utils import LOGGER, colorstr
import argparse
import yaml  
# from ultralytics.data.utils import check_dataset # (V8.2 移除)
from skimage.metrics import structural_similarity as ssim

from skimage.metrics import peak_signal_noise_ratio as psnr

# -------------------------------------------------------------------
# [!!! V8.4 修复 !!!]
# -------------------------------------------------------------------
class SharedGradCAM:
    """
    Grad-CAM implementation for Shared-Backbone models that
    process (rgb, ir) tuples and output (feat_rgb, feat_ir) as
    *two separate return values* from their forward pass.
    """
    
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
    
    # [!!! 关键修复 V8.4 !!!]
    def save_gradient(self, layer_name):
        """
        Save gradient hook.
        'grad_output' is a tuple (grad_rgb, grad_ir) because the
        module's forward pass returned two tensors.
        """
        def hook(module, grad_input, grad_output):
            # grad_output 应该是一个 (grad_rgb, grad_ir) 元组
            if isinstance(grad_output, (tuple, list)) and len(grad_output) == 2:
                # 检查梯度是否存在 (可能一个分支没有用于loss)
                if grad_output[0] is not None:
                    self.gradients[f'{layer_name}_rgb'] = grad_output[0].detach()
                if grad_output[1] is not None:
                    self.gradients[f'{layer_name}_ir'] = grad_output[1].detach()
            
            # [!!! 新增调试 !!!] 
            # 帮助你看到梯度的 *确切* 格式
            else:
                LOGGER.warning(f"DEBUG: grad_output at {layer_name} is NOT a 2-tuple.")
                LOGGER.warning(f"DEBUG: Type: {type(grad_output)}")
                if isinstance(grad_output, (tuple, list)):
                    LOGGER.warning(f"DEBUG: Len: {len(grad_output)}")
                    # 也许它是一个 (grad_tensor,) 元组？
                    if len(grad_output) == 1 and grad_output[0] is not None:
                         LOGGER.warning(f"DEBUG: grad_output[0] type: {type(grad_output[0])}")
                         # 这不应该发生，但作为最后的防线
                         # self.gradients[f'{layer_name}_rgb'] = grad_output[0].detach()
                         # self.gradients[f'{layer_name}_ir'] = grad_output[0].detach()
                
        return hook
    
    def save_activation(self, layer_name):
        """
        Save activation hook.
        'output' is a tuple (feat_rgb, feat_ir) because the
        module's forward pass returned two tensors.
        """
        def hook(module, input, output):
            # output 应该是一个 (feat_rgb, feat_ir) 元组
            if isinstance(output, (tuple, list)) and len(output) == 2:
                self.activations[f'{layer_name}_rgb'] = output[0].detach()
                self.activations[f'{layer_name}_ir'] = output[1].detach()
            else:
                LOGGER.warning(f"Unexpected activation format at {layer_name}. Expected 2-tuple (feat_rgb, feat_ir).")

        return hook
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        for layer_idx in self.target_layers:
            try:
                layer = self.model.model.model[layer_idx]
                
                # Forward hook
                handle_forward = layer.register_forward_hook(
                    self.save_activation(f'layer_{layer_idx}')
                )
                self.hooks.append(handle_forward)
                
                # Backward hook
                handle_backward = layer.register_full_backward_hook(
                    self.save_gradient(f'layer_{layer_idx}')
                )
                self.hooks.append(handle_backward)
            except Exception as e:
                LOGGER.warning(f"Failed to register hook for layer {layer_idx}. It may be an invalid layer. Error: {e}")
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_tuple):
        """
        Generate CAMs for ALL target layers in one pass, using tuple input.
        """
        # --- 1. FORWARD PASS ---
        for param in self.model.model.parameters():
            param.requires_grad = True
        self.model.model.zero_grad()
        
        with torch.enable_grad():
            # [!!! V8.3 修复 !!!] 使用 *解包后* 的 tuple input
            # 假设模型 forward 是 def forward(self, x_rgb, x_ir)
            output = self.model.model(*input_tuple)
        
        # --- 2. BACKWARD PASS ---
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        score = output.max()
        score.backward() # 只反向传播一次

        # --- 3. GENERATE ALL CAMS ---
        cam_dict = {}
        for layer_idx in self.target_layers:
            layer_name = f'layer_{layer_idx}'
            
            # --- Process RGB CAM ---
            grad_rgb = self.gradients.get(f'{layer_name}_rgb')
            act_rgb = self.activations.get(f'{layer_name}_rgb')
            
            if grad_rgb is not None and act_rgb is not None:
                weights_rgb = torch.mean(grad_rgb, dim=(2, 3), keepdim=True)
                cam_rgb = torch.sum(weights_rgb * act_rgb, dim=1, keepdim=True)
                cam_rgb = F.relu(cam_rgb)
                
                cam_rgb = cam_rgb.squeeze().cpu().numpy()
                cam_rgb_min, cam_rgb_max = cam_rgb.min(), cam_rgb.max()
                cam_dict[f'{layer_idx}_rgb'] = (cam_rgb - cam_rgb_min) / (cam_rgb_max - cam_rgb_min + 1e-8)
            else:
                # [!!!] V8.4 - 仅在V8.3修复后仍失败时才打印
                if not (grad_rgb is None and act_rgb is None):
                    LOGGER.warning(f"Missing RGB gradients OR activations for {layer_name}")

            # --- Process IR CAM ---
            grad_ir = self.gradients.get(f'{layer_name}_ir')
            act_ir = self.activations.get(f'{layer_name}_ir')
            
            if grad_ir is not None and act_ir is not None:
                weights_ir = torch.mean(grad_ir, dim=(2, 3), keepdim=True)
                cam_ir = torch.sum(weights_ir * act_ir, dim=1, keepdim=True)
                cam_ir = F.relu(cam_ir)
                
                cam_ir = cam_ir.squeeze().cpu().numpy()
                cam_ir_min, cam_ir_max = cam_ir.min(), cam_ir.max()
                cam_dict[f'{layer_idx}_ir'] = (cam_ir - cam_ir_min) / (cam_ir_max - cam_ir_min + 1e-8)
            else:
                 if not (grad_ir is None and act_ir is None):
                    LOGGER.warning(f"Missing IR gradients OR activations for {layer_name}")
        
        # [!!! V8.4 !!!] 增加检查，如果字典为空则警告
        if not cam_dict:
             LOGGER.warning(f"Failed to generate any CAMs. Gradients or Activations were missing for all layers.")
             
        return cam_dict
# -------------------------------------------------------------------
# [END NEW CLASS]
# -------------------------------------------------------------------


# --- 辅助函数，用于打印 SSIM 报告 (无需修改) ---
def log_average_ssim(similarity_results, image_count, is_final=False):
    """
    计算并记录 similarity_results 列表中的平均 SSIM 分数。
    """
    if not similarity_results:
        LOGGER.info("No similarity results to report.")
        return

    # 计算平均值
    avg_p1 = np.mean([res['sim_P1'] for res in similarity_results])
    avg_p2 = np.mean([res['sim_P2'] for res in similarity_results])
    avg_p3 = np.mean([res['sim_P3'] for res in similarity_results])
    avg_p4 = np.mean([res['sim_P4'] for res in similarity_results])
    avg_p5 = np.mean([res['sim_P5'] for res in similarity_results])

    # 格式化标题
    if is_final:
        header = "Similarity analysis completed!"
        analysis_text = f"Analyzed {image_count} images."
    else:
        header = f"--- Interim Report: Average SSIM after {image_count} images ---"
        analysis_text = ""

    LOGGER.info(f"{colorstr('green', 'bold', header)}")
    if analysis_text:
        LOGGER.info(analysis_text)
        
    LOGGER.info(f"  Average P1 SSIM: {avg_p1:.4f}")
    LOGGER.info(f"  Average P2 SSIM: {avg_p2:.4f}")
    LOGGER.info(f"  Average P3 SSIM: {avg_p3:.4f}")
    LOGGER.info(f"  Average P4 SSIM: {avg_p4:.4f}")
    LOGGER.info(f"  Average P5 SSIM: {avg_p5:.4f}")
    
    if not is_final:
        LOGGER.info(f"-------------------------------------------------------")
# --- [END 辅助函数] ---


def visualize_shared_backbone_cam(
    model_path,
    data_yaml,
    save_dir='runs/cam_visualization_shared',
    batch_size=1,
    imgsz=640,
    max_images=None,
    device='cuda:0'
):
    """
    [MODIFIED V8.2]
    适配 'build_yolo_dataset_m' 自定义数据加载器。
    移除了 'check_dataset' 依赖。
    """
    # --- 1. 文件夹设置 (无需修改) ---
    save_dir = Path(save_dir)
    comparison_dir = save_dir / 'comparison'
    p1_dir = save_dir / 'P1'
    p2_dir = save_dir / 'P2'
    p3_dir = save_dir / 'P3'
    p4_dir = save_dir / 'P4'
    p5_dir = save_dir / 'P5'
    
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (p1_dir / 'RGB').mkdir(parents=True, exist_ok=True)
    (p1_dir / 'IR').mkdir(parents=True, exist_ok=True)
    (p2_dir / 'RGB').mkdir(parents=True, exist_ok=True)
    (p2_dir / 'IR').mkdir(parents=True, exist_ok=True)
    (p3_dir / 'RGB').mkdir(parents=True, exist_ok=True)
    (p3_dir / 'IR').mkdir(parents=True, exist_ok=True)
    (p4_dir / 'RGB').mkdir(parents=True, exist_ok=True)
    (p4_dir / 'IR').mkdir(parents=True, exist_ok=True)
    (p5_dir / 'RGB').mkdir(parents=True, exist_ok=True)
    (p5_dir / 'IR').mkdir(parents=True, exist_ok=True)

    stage_names = ['Stage 1 (P1)', 'Stage 2 (P2)', 'Stage 3 (P3)', 'Stage 4 (P4)', 'Stage 5 (P5)'] # 对应 P1-P5
    stage_dir_map = {
        'Stage 1 (P1)': p1_dir,
        'Stage 2 (P2)': p2_dir,
        'Stage 3 (P3)': p3_dir,
        'Stage 4 (P4)': p4_dir,
        'Stage 5 (P5)': p5_dir
    }
    
    # --- 2. 加载模型与数据 ---
    LOGGER.info(f"Loading model from {model_path}")
    model = YOLO(model_path)
    model.model.to(device)
    model.model.eval()
    
    # [!!!] 目标图层 (根据你的 V8.0 YAML)
    target_layers = [1, 3, 5, 7, 10] 
    
    grad_cam = SharedGradCAM(model, target_layers)
    grad_cam.register_hooks()
    
    # -----------------------------------------------------------------
    # [!!! 关键修复 V8.2 !!!] - 修复数据加载 (移除 check_dataset)
    # -----------------------------------------------------------------
    LOGGER.info(f"Loading validation dataset from {data_yaml} using custom 'build_yolo_dataset_m'")
    
    # --- Imports for data loading ---
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    
    # --- Setup config args ---
    args = get_cfg(DEFAULT_CFG)
    args.data = data_yaml
    args.imgsz = imgsz
    args.batch = batch_size
    args.workers = 0 # 调试时建议设为0
    
    # --- [V8.2] 手动解析 YAML ---
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # 1. 手动获取 'path'
    path = Path(data.get('path', '')) 
    
    # 2. 手动获取 'names' 和 'nc'
    if 'names' not in data:
        LOGGER.error(f"Fatal: 你的数据 YAML '{data_yaml}' 缺少 'names' 键。")
        return
    names = data['names']
    nc = len(names)

    # 3. 手动获取路径
    try:
        val_path_rgb = str(path / data['val_rgb'])
        val_path_ir = str(path / data['val_ir'])
    except KeyError:
        LOGGER.error(f"Fatal: 你的数据 YAML '{data_yaml}' 缺少 'val_rgb' 或 'val_ir' 键。")
        LOGGER.error("新模型的数据加载器需要独立的 RGB 和 IR 验证路径。")
        return

    # 4. 将 'names' 和 'nc' 放入 data 字典 (YOLODataset_m 可能需要)
    #    并放入 args (build_yolo_dataset_m 需要)
    data['names'] = names
    data['nc'] = nc
    args.classes = names
    
    # --- [!!! 关键 !!!] ---
    # 构建自定义的 dataset
    LOGGER.info(f"Building dataset 'YOLODataset_m'...")
    dataset = build_yolo_dataset_m(
        cfg=args,
        img_path_rgb=val_path_rgb,   # <--- 传递 *路径*
        img_path_ir=val_path_ir,    # <--- 传递 *路径*
        batch=batch_size,           # <--- 传递 *batch_size*
        data=data,                  # <--- 传递 data 字典
        mode='val',
        stride=32
    )
    
    dataloader = build_dataloader(dataset, batch_size, args.workers, shuffle=False, rank=-1) 
    # -----------------------------------------------------------------
    # [!!! 数据加载修改结束 !!!]
    # -----------------------------------------------------------------

    
    # --- 3. 处理图像 ---
    num_images = len(dataset) if (max_images is None or max_images == 0) else min(max_images, len(dataset))
    LOGGER.info(f"Processing {num_images} images...")
    
    similarity_results = []
    
    processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, total=num_images)):
        if max_images is not None and max_images > 0 and processed >= max_images:
            break
        
        try:
            img_rgb = batch['img_rgb'].to(device)
            img_ir = batch['img_ir'].to(device)
            img_name = Path(batch['im_file_rgb'][0]).stem # 假设 'im_file_rgb' 存在
        except KeyError as e:
            LOGGER.error(f"Fatal: Dataloader 返回的 'batch' 字典中缺少键: {e}。")
            LOGGER.error("请确认 YOLODataset_m 的 __getitem__ 和 collate_fn 返回了 'img_rgb', 'img_ir', 和 'im_file_rgb'。")
            break
        
        # 准备模型输入 (Tuple)
        img_rgb_tensor = img_rgb.float() / 255.0
        img_ir_tensor = img_ir.float() / 255.0
        input_tuple = (img_rgb_tensor, img_ir_tensor)

        # 准备可视化原图 (BGR)
        img_rgb_vis_rgb = img_rgb[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        img_ir_vis_orig = img_ir[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        
        if img_ir_vis_orig.shape[2] == 1:
            img_ir_vis_bgr = cv2.cvtColor(img_ir_vis_orig, cv2.COLOR_GRAY2BGR)
        else:
            # 假设 3-channel IR 是 RGB 格式
            img_ir_vis_bgr = cv2.cvtColor(img_ir_vis_orig, cv2.COLOR_RGB2BGR)
        
        # --- 4. 高效CAM生成 ---
        cams = grad_cam.generate_cam(input_tuple)
        
        if not cams:
             LOGGER.warning(f"Failed to generate CAMs for {img_name}, skipping.")
             continue
             
        # --- 5. P1-P5 全局相似度计算 ---
        sim_p1, sim_p2, sim_p3, sim_p4, sim_p5 = 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            if f'{target_layers[0]}_rgb' in cams and f'{target_layers[0]}_ir' in cams:
                sim_p1 = ssim(cams[f'{target_layers[0]}_rgb'], cams[f'{target_layers[0]}_ir'], data_range=1.0)
            if f'{target_layers[1]}_rgb' in cams and f'{target_layers[1]}_ir' in cams:
                sim_p2 = ssim(cams[f'{target_layers[1]}_rgb'], cams[f'{target_layers[1]}_ir'], data_range=1.0)
            if f'{target_layers[2]}_rgb' in cams and f'{target_layers[2]}_ir' in cams:
                sim_p3 = ssim(cams[f'{target_layers[2]}_rgb'], cams[f'{target_layers[2]}_ir'], data_range=1.0)
            if f'{target_layers[3]}_rgb' in cams and f'{target_layers[3]}_ir' in cams:
                sim_p4 = ssim(cams[f'{target_layers[3]}_rgb'], cams[f'{target_layers[3]}_ir'], data_range=1.0)
            if f'{target_layers[4]}_rgb' in cams and f'{target_layers[4]}_ir' in cams:
                sim_p5 = ssim(cams[f'{target_layers[4]}_rgb'], cams[f'{target_layers[4]}_ir'], data_range=1.0)
        except Exception as e:
            LOGGER.warning(f"SSIM calculation failed for {img_name}: {e}")

        similarity_results.append({
            "image_name": img_name,
            "sim_P1": sim_p1,
            "sim_P2": sim_p2,
            "sim_P3": sim_p3,
            "sim_P4": sim_p4,
            "sim_P5": sim_p5
        })
        
        # --- 6. [MODIFIED V8.0] 可视化与保存 (5x4 Grid, 叠加图, 基于IR图) ---
        fig = plt.figure(figsize=(15, 18))
        num_stages = 5
        num_cols = 4 # 4 列: (IR-Orig, RGB-Overlay-on-IR | IR-Orig, IR-Overlay-on-IR)
        
        for stage_idx in range(num_stages):
            target_layer = target_layers[stage_idx]
            current_stage_dir = stage_dir_map[stage_names[stage_idx]]
            row = stage_idx
            
            # --- RGB visualizations (Base on IR) ---
            
            # IR Original (Base)
            ax = plt.subplot(num_stages, num_cols, row * num_cols + 1)
            if img_ir_vis_orig.shape[2] == 1:
                ax.imshow(img_ir_vis_orig.squeeze(), cmap='gray')
            else:
                ax.imshow(img_ir_vis_orig)
            ax.set_title(f'{stage_names[stage_idx]} - IR Original (Base)', fontsize=10)
            ax.axis('off')

            # RGB Overlay (on IR)
            cam_key_rgb = f'{target_layer}_rgb'
            if cam_key_rgb in cams:
                cam_rgb = cams[cam_key_rgb]
                heatmap_rgb_bgr = cv2.applyColorMap(np.uint8(255 * cam_rgb), cv2.COLORMAP_JET)
                heatmap_rgb_bgr = cv2.resize(heatmap_rgb_bgr, (img_ir_vis_bgr.shape[1], img_ir_vis_bgr.shape[0]))
                overlay_rgb_bgr = cv2.addWeighted(heatmap_rgb_bgr, 0.5, img_ir_vis_bgr, 0.5, 0)
                rgb_save_path = current_stage_dir / 'RGB' / f'{img_name}.png'
                cv2.imwrite(str(rgb_save_path), overlay_rgb_bgr)
                overlay_rgb_rgb = cv2.cvtColor(overlay_rgb_bgr, cv2.COLOR_BGR2RGB)
                ax = plt.subplot(num_stages, num_cols, row * num_cols + 2)
                ax.imshow(overlay_rgb_rgb)
                ax.set_title(f'RGB Overlay on IR (L{target_layer})', fontsize=10)
                ax.axis('off')
            
            # --- IR visualizations ---
            
            # IR Original
            ax = plt.subplot(num_stages, num_cols, row * num_cols + 3)
            if img_ir_vis_orig.shape[2] == 1:
                ax.imshow(img_ir_vis_orig.squeeze(), cmap='gray')
            else:
                ax.imshow(img_ir_vis_orig)
            ax.set_title(f'{stage_names[stage_idx]} - IR Original', fontsize=10)
            ax.axis('off')
            
            # IR Overlay
            cam_key_ir = f'{target_layer}_ir'
            if cam_key_ir in cams:
                cam_ir = cams[cam_key_ir]
                heatmap_ir_bgr = cv2.applyColorMap(np.uint8(255 * cam_ir), cv2.COLORMAP_JET)
                heatmap_ir_bgr = cv2.resize(heatmap_ir_bgr, (img_ir_vis_bgr.shape[1], img_ir_vis_bgr.shape[0]))
                overlay_ir_bgr = cv2.addWeighted(heatmap_ir_bgr, 0.5, img_ir_vis_bgr, 0.5, 0)
                ir_save_path = current_stage_dir / 'IR' / f'{img_name}.png'
                cv2.imwrite(str(ir_save_path), overlay_ir_bgr)
                overlay_ir_rgb = cv2.cvtColor(overlay_ir_bgr, cv2.COLOR_BGR2RGB)
                ax = plt.subplot(num_stages, num_cols, row * num_cols + 4)
                ax.imshow(overlay_ir_rgb)
                ax.set_title(f'IR Overlay (L{target_layer})', fontsize=10)
                ax.axis('off')

        fig.suptitle(f'Shared-Backbone Grad-CAM Visualization - {img_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = comparison_dir / f'{img_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"Saved visualizations for {img_name}")
        
        # --- 周期性报告 (无需修改) ---
        current_image_count = processed + 1
        if current_image_count % 10 == 0 and similarity_results:
            log_average_ssim(similarity_results, current_image_count, is_final=False)

        processed += 1
    
    # --- 7. 保存并打印 *最终* 相似度报告 ---
    grad_cam.remove_hooks()
    
    # [V8.4] 增加检查，防止在所有图像都失败时出错
    if not similarity_results:
        LOGGER.error("No similarity results generated for any image. CAM generation failed completely.")
        return

    json_save_path = save_dir / 'similarity_results.json'
    with open(json_save_path, 'w') as f:
        json.dump(similarity_results, f, indent=2)
    
    log_average_ssim(similarity_results, len(similarity_results), is_final=True)
            
    LOGGER.info(f"Full similarity results saved to: {json_save_path.resolve()}")
    LOGGER.info(f"Visualizations saved to {save_dir.resolve()}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Shared-Backbone Grad-CAM Visualization (V8.4)')
    
    parser.add_argument('--model', type=str, required=True, help='Path to the NEW shared-backbone model .pt file')
    parser.add_argument('--data', type=str, required=True, help='Path to the data.yaml file (must contain val_rgb and val_ir)')
    parser.add_argument('--save-dir', type=str, default='runs/cam_visualization_shared', help='Save directory')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (must be 1 for CAM)')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process (0 or None for all)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # [MODIFIED] 调用主函数
    visualize_shared_backbone_cam(
        model_path=args.model,
        data_yaml=args.data,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        max_images=args.max_images,
        device=args.device
    )


if __name__ == '__main__':
    main()