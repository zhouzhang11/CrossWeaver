# 验证可视化功能使用指南

## 概述

本指南说明如何在验证过程中将每张图片的预测检测框绘制到原图上并保存到输出文件夹。

## 修改内容

### 1. 修改的文件

- `new/ultralytics/engine/validator_m.py` - 基础验证器类
- `new/ultralytics/models/yolo/obb/val_m.py` - OBB验证器类
- `new/val_m.py` - 验证脚本

### 2. 新增功能

#### 2.1 添加 `save_vis` 参数

在验证时可以使用 `save_vis=True` 参数来启用可视化保存功能。

#### 2.2 输出目录结构

```
<project>/<name>/out/
├── rgb/          # RGB图像的可视化结果
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ir/           # 红外图像的可视化结果
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## 使用方法

### 方法 1: 使用 val_m.py

直接运行修改后的 `val_m.py`:

```bash
cd /home/sysspace/yanght/test/new
python val_m.py
```

### 方法 2: 自定义脚本

```python
from ultralytics import YOLO_m

# 加载模型
model = YOLO_m('/path/to/your/weights/best.pt')

# 运行验证并保存可视化
model.val(
    data='M3FD.yaml',
    project='output_project',
    name='experiment_name',
    imgsz=800,
    batch=16,
    device=0,
    save_vis=True,  # 启用可视化保存
)
```

### 方法 3: 使用测试脚本

```bash
cd /home/sysspace/yanght/test/new
python test_visualization.py
```

## 可视化特性

### 1. 绘制内容

- ✓ 旋转检测框（OBB）
- ✓ 类别标签
- ✗ 置信度分数（按要求不显示）

### 2. 颜色方案

- 每个类别使用固定的颜色
- 颜色自动从预定义的调色板中选择
- 确保不同类别之间有良好的视觉区分度

### 3. 图像格式

- 保持原始图像分辨率
- 使用原始文件名保存
- 支持 RGB 和 IR 双模态图像

## 技术细节

### 1. 坐标转换

预测框会自动从模型输出坐标系转换回原始图像坐标系，确保检测框准确对齐。

### 2. 旋转框绘制

使用旋转矩阵计算旋转框的四个角点坐标：

```python
# xywhr 格式转换为四个角点
corners = [
    [-w/2, -h/2],
    [w/2, -h/2],
    [w/2, h/2],
    [-w/2, h/2]
]
# 应用旋转和平移
rotated_corners = corners @ rotation_matrix.T + center
```

### 3. 性能考虑

- 可视化保存会增加验证时间
- 建议在需要时才启用 `save_vis=True`
- 批量处理时注意内存管理

## 输出示例

运行验证后，你会在指定的输出目录看到：

```
DroneVehicle/val_yolov8l-obb_adalora_sym_m_r9-6-wop_e50_bs8_best/
├── out/
│   ├── rgb/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   └── ir/
│       ├── 00001.jpg
│       ├── 00002.jpg
│       └── ...
├── labels/
├── confusion_matrix.png
└── results.csv
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_vis` | bool | False | 是否保存可视化结果 |
| `data` | str | - | 数据集配置文件 |
| `project` | str | 'runs/detect' | 项目保存目录 |
| `name` | str | 'exp' | 实验名称 |
| `imgsz` | int | 640 | 输入图像大小 |
| `batch` | int | 16 | 批次大小 |
| `device` | int/str | 0 | 设备ID |
| `conf` | float | 0.001 | 置信度阈值 |
| `iou` | float | 0.6 | NMS IoU阈值 |

## 故障排除

### 问题 1: 输出目录未创建

**原因**: `save_vis` 参数未设置或设置为 False

**解决**: 确保在调用 `model.val()` 时设置 `save_vis=True`

### 问题 2: 图像无法读取

**原因**: 图像路径不正确或文件不存在

**解决**: 检查数据集配置文件中的路径设置

### 问题 3: 检测框位置不准确

**原因**: 坐标转换问题

**解决**: 这个问题已在代码中处理，使用 `ops.scale_boxes()` 进行正确的坐标转换

## 注意事项

1. **内存使用**: 大批次验证时注意内存占用
2. **磁盘空间**: 确保有足够的磁盘空间保存可视化图像
3. **文件命名**: 使用原始文件名，避免重名覆盖
4. **性能影响**: 启用可视化会增加验证时间，建议在需要时使用

## 更新日志

- 2025-01-10: 初始版本，实现基础可视化功能
- 移除置信度显示，仅显示类别标签
- 支持 RGB 和 IR 双模态图像
- 自动创建输出目录结构

## 联系方式

如有问题或建议，请联系开发团队。
