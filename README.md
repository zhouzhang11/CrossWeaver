# CrossWeaver

> **CrossWeaver: Cross-Modal Feature Weaving with Adaptive LoRA for Aerial Object Detection**
>
> CVPR 2026

The official PyTorch implementation of CrossWeaver, a cross-modal (RGB + Infrared) object detection framework that weaves homogeneous and heterogeneous features through Adaptive LoRA and Mamba-based fusion for robust aerial detection.

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).

## Overview

CrossWeaver addresses the challenge of fusing RGB and infrared (IR) features for aerial object detection. The key idea is to **separate** what is shared (homogeneous) from what is modality-specific (heterogeneous), then adaptively **weave** them back together.

<div align="center">
  <img src="docs/architecture.png" width="800"/>
</div>

### Key components

- **Homogeneous / Heterogeneous Feature Separation** — A dual-path backbone extracts modality-shared (homo) and modality-specific (het) features from RGB–IR pairs, guided by a Prior Extraction module.
- **Adaptive LoRA (AdaLoRA)** — Symmetric low-rank adaptation with dynamically pruned ranks tailors the pre-trained YOLOv8 backbone to cross-modal inputs without full fine-tuning.
- **Mamba Fusion** — A state-space model (Mamba) fuses homo features at the deepest level, capturing long-range cross-modal dependencies efficiently.
- **DCN Alignment** — Deformable convolution warps multi-scale features to handle spatial misalignment between modalities.
- **Dual Detection Heads** — Separate detection branches for RGB and IR produce the final oriented bounding boxes.

## Installation

```bash
# Clone the repository
git clone https://github.com/zhouzhang11/CrossWeaver.git
cd CrossWeaver

# Install dependencies
pip install -e .
pip install einops causal-conv1d mamba-ssm
```

**Requirements:** Python >= 3.8, PyTorch >= 1.8, CUDA (for Mamba/selective scan kernels)

## Datasets

Download the datasets from their official sources:

| Dataset | Description | Link |
|---------|-------------|------|
| DroneVehicle | RGB-IR vehicle detection with OBB | [Download](https://drive.google.com/drive/folders/1v91V7bcr6Hu6gRzDJ1zK7_YKOJdaxaBA) |
| M3FD | Multi-modal multi-scale fusion detection | [Official]() |
| DVTOD | Drone-view thermal object detection | [Official]() |
| VEDAI | Vehicle detection in aerial imagery | [Official]() |

Organize datasets following the Ultralytics YOLO format. Dataset YAML configuration files are in `ultralytics/cfg/datasets/`.

## Quick Start

### Training

Standard YOLOv8 baseline:
```bash
python train.py
```

CrossWeaver (AdaLoRA + Mamba + DCN):
```bash
python train_m.py
```

Key training arguments:
- `r_init` / `r_target` — Initial and target LoRA ranks for AdaLoRA
- `adalora` — Enable adaptive rank pruning
- `data` — Dataset YAML config path
- `model` — Model config YAML (see `ultralytics/cfg/models/lma/`)

### Validation

```bash
python val.py          # Standard
python val_m.py        # CrossWeaver
python val_shift.py    # Spatial shift robustness evaluation
```

### Testing

```bash
bash test.sh                  # Full test suite
bash test_single_shift.sh     # Single shift test
```

## Model Zoo

Model configurations are in `ultralytics/cfg/models/lma/`:

| Config | Description |
|--------|-------------|
| `cvpr.yaml` | Full CrossWeaver (homo/het paths, AdaLoRA sym, Mamba fusion, DCN) |
| `yolov8_m.yaml` | YOLOv8 multi-modal variant |
| `yolov8_lma.yaml` | YOLOv8 with LoRA modules |
| `yolov8-obb_m.yaml` | OBB variant for oriented detection |
| `yolov8-obb_lma.yaml` | OBB variant with LoRA |
| `new.yaml`, `newnew.yaml` | Experimental variants |

## Repository Structure

```
.
├── train_m.py                           # CrossWeaver training entry
├── val_m.py                             # CrossWeaver validation entry
├── val_shift.py                         # Spatial shift robustness evaluation
├── train.py / val.py                    # YOLOv8 baseline
├── convert.py                           # Model conversion utilities
├── ssm/                                 # Mamba / Selective Scan kernels
├── ultralytics/
│   ├── nn/modules/
│   │   ├── conv_adalora_symmetric_m.py  # AdaLoRA symmetric (multi-modal)
│   │   ├── conv_adalora_asymmetric_m.py # AdaLoRA asymmetric
│   │   ├── conv_adalora.py             # AdaLoRA base
│   │   ├── conv_adalora_homo_het.py    # Homo/het LoRA variants
│   │   ├── conv_lora.py / conv_lora_m.py
│   │   ├── conv_all_lora.py / conv_all_lora_m.py
│   │   └── como_block.py               # Fusion & prior injection blocks
│   ├── engine/
│   │   ├── trainer_m.py / validator_m.py
│   │   ├── adalora_rank_allocator.py    # Adaptive rank pruning
│   │   └── trainer.py / validator.py
│   ├── nn/tasks_m.py                    # Multi-modal task definition
│   └── cfg/models/lma/                  # Model configs
└── test_modules.py                      # Module-level unit tests
```

## Citation

If you use CrossWeaver in your research, please cite:

```bibtex
@inproceedings{crossweaver2026,
  title     = {CrossWeaver: Cross-Modal Feature Weaving with Adaptive LoRA for Aerial Object Detection},
  author    = {},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

## License

This project is released under the [AGPL-3.0 License](LICENSE), following the Ultralytics YOLOv8 license.

## Acknowledgements

This codebase is built upon [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). The SSM/Mamba kernels are adapted from [Mamba](https://github.com/state-spaces/mamba) by Tri Dao and Albert Gu.
