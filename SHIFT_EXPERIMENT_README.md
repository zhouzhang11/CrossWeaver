# Shift Robustness Experiment

本文档说明如何运行和分析RGB-IR图像偏移鲁棒性实验。

## 实验目的

评估多模态IR-RGB目标检测模型对空间错位的鲁棒性。通过在验证集上对RGB图像施加不同程度的空间偏移（x和y方向各[-30, 30]像素，间隔5），测试模型性能下降情况。

## 关键设计

- **IR图像保持不变**
- **RGB图像进行偏移**（实时计算，不创建物理副本）
- **标注保持不变**（模拟真实的配准误差场景）
- **偏移后空白区域填充黑色**

## 文件说明

### 核心文件

1. **base_m.py** (已修改)
   - 添加了 `shift_x` 和 `shift_y` 参数
   - 实现了 `apply_shift()` 函数用于图像偏移
   - 在 `load_image_ir()` 中应用偏移

2. **val_shift.py** (新建)
   - 支持命令行参数的验证脚本
   - 可指定偏移量、GPU设备等参数

3. **generate_shift_scripts.py** (新建)
   - 生成4个GPU的运行脚本
   - 自动分配168个实验到4张卡

4. **collect_shift_results.py** (新建)
   - 收集所有实验结果
   - 生成CSV、热力图和统计分析

## 使用步骤

### 步骤1: 生成运行脚本

```bash
cd /home/sysspace/yanght/test/new
python generate_shift_scripts.py
```

这将生成以下文件：
- `run_shift_gpu0.sh` - GPU 0的运行脚本（42个任务）
- `run_shift_gpu1.sh` - GPU 1的运行脚本（42个任务）
- `run_shift_gpu2.sh` - GPU 2的运行脚本（42个任务）
- `run_shift_gpu3.sh` - GPU 3的运行脚本（42个任务）
- `run_all_shifts.sh` - 主脚本（并行运行所有GPU）

### 步骤2: 运行实验

**方式1 - 并行运行所有GPU（推荐）：**

```bash
cd /home/sysspace/yanght/test/new
./run_all_shifts.sh
```

**方式2 - 在4个终端中分别运行：**

```bash
# Terminal 1
cd /home/sysspace/yanght/test/new && ./run_shift_gpu0.sh

# Terminal 2
cd /home/sysspace/yanght/test/new && ./run_shift_gpu1.sh

# Terminal 3
cd /home/sysspace/yanght/test/new && ./run_shift_gpu2.sh

# Terminal 4
cd /home/sysspace/yanght/test/new && ./run_shift_gpu3.sh
```

### 步骤3: 监控进度

查看日志文件：

```bash
# 实时查看GPU 0的进度
tail -f shift_gpu0.log

# 查看所有GPU的进度
tail -f shift_gpu*.log
```

### 步骤4: 收集和分析结果

实验完成后，运行：

```bash
cd /home/sysspace/yanght/test/new
python collect_shift_results.py
```

这将生成：
- `DroneVehicle/shift_analysis/shift_results.csv` - 所有结果的CSV文件
- `DroneVehicle/shift_analysis/heatmap_*.png` - 各指标的热力图
- `DroneVehicle/shift_analysis/statistics.txt` - 统计分析报告

## 实验配置

### 当前配置

- **模型权重**: `/home/sysspace/yanght/test/new/DroneVehicle/yolov8s_new_model10/weights/best.pt`
- **数据配置**: `drone_vehicle_m.yaml`
- **图像大小**: 800
- **批次大小**: 16
- **偏移范围**: x ∈ [-30, 30], y ∈ [-30, 30]
- **偏移间隔**: 5像素
- **总实验数**: 168 (13×13 - 1，排除(0,0))

### 修改配置

如需修改配置，编辑 `generate_shift_scripts.py` 中的以下变量：

```python
WEIGHTS = '...'      # 模型权重路径
DATA_YAML = '...'    # 数据配置文件
IMGSZ = 800          # 图像大小
BATCH = 16           # 批次大小
```

然后重新运行步骤1生成新的脚本。

## 单次测试

如果想单独测试某个偏移量：

```bash
python val_shift.py --shift_x 5 --shift_y 10 --device 0
```

参数说明：
- `--shift_x`: X方向偏移（正数=向右，负数=向左）
- `--shift_y`: Y方向偏移（正数=向下，负数=向上）
- `--device`: GPU设备ID
- `--weights`: 模型权重路径（可选）
- `--data`: 数据配置文件（可选）
- `--imgsz`: 图像大小（可选）
- `--batch`: 批次大小（可选）

## 结果解读

### CSV文件

包含每个偏移组合的详细指标：
- `shift_x`, `shift_y`: 偏移量
- `mAP50`, `mAP50-95`: 平均精度
- `precision`, `recall`: 精确率和召回率

### 热力图

- **颜色**: 绿色=高性能，红色=低性能
- **X轴**: X方向偏移（正=右，负=左）
- **Y轴**: Y方向偏移（正=下，负=上）
- **中心**: 通常对应(0,0)无偏移的基准性能

### 统计报告

包含：
- 各指标的均值、标准差、最大值、最小值
- 最佳和最差性能的偏移量
- 相对于基准(0,0)的性能下降百分比

## 预期时间

- 单次验证: 约2-5分钟（取决于硬件）
- 总实验时间: 约3-8小时（4张GPU并行）

## 故障排除

### 问题1: 找不到模块

```bash
# 确保在正确的环境中
conda activate your_env
```

### 问题2: GPU内存不足

减小批次大小：
```bash
# 编辑 generate_shift_scripts.py
BATCH = 8  # 从16改为8
```

### 问题3: 结果收集失败

检查结果目录是否存在：
```bash
ls DroneVehicle/shift_experiments/
```

### 问题4: 权重文件不存在

确认权重文件路径：
```bash
ls /home/sysspace/yanght/test/new/DroneVehicle/yolov8s_new_model10/weights/best.pt
```

## 注意事项

1. **磁盘空间**: 每个实验会生成约10-50MB的结果文件，总共约2-8GB
2. **GPU显存**: 确保每张GPU有足够显存（建议≥8GB）
3. **数据路径**: 确保 `drone_vehicle_m.yaml` 中的路径正确
4. **标注不变**: 代码已确保标注不会随图像偏移而改变

## 联系方式

如有问题，请联系项目维护者。
