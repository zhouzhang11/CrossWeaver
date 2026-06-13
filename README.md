# CrossWeaver: Towards Efficient Cross-Modal Interweaving and Decoupling for Weakly-Aligned Multispectral Object Detection

[![Conference](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://cvpr.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation of the CVPR 2026 Findings paper **"CrossWeaver: Towards Efficient Cross-Modal Interweaving and Decoupling for Weakly-Aligned Multispectral Object Detection"**.

> **Note:** The code is currently undergoing internal review and clean-up. We will release the full training and evaluation code, along with the pre-trained weights, before the conference starts. 

## 📢 News / Updates
* **[2026-04-01]** Our paper has been accepted to **CVPR 2026 Findings**!
* **[2026-04-01]** Created the repository. The source code and pre-trained models will be released soon.

## 🛠️ TODO List
- [x] Initial repository setup.
- [ ] Release core network architecture code (`models/CrossWeaver.py`).
- [ ] Release training scripts and dataloaders for the DroneVehicle dataset.
- [ ] Release pre-trained weights.
- [ ] Provide evaluation instructions.

## 📖 Abstract
While single-modal object detection has made significant progress, real-world perception increasingly depends on multimodal data to capture richer visual cues. However, multimodal object detection under weak spatial alignment remains challenging. In this work, we revisit the problem from two key perspectives: (1) quantization error caused by misaligned multimodal fusion, and (2) balancing modality similarity and complementarity for robust feature integration. Building upon these insights, we propose a novel Cross-Modal Semantic Weaving Network (CrossWeaver) for weakly aligned multimodal object detection. Specifically, we design the Modality-Contrastive Shared Encoder (MCSE) to extract shared semantic representations across modalities under contrastive supervision. Then, we propose the Modality-Conditioned Deformation Adapter (MCDA) to dynamically modulate spatial sampling fields for adaptive geometric transformations and inject lightweight modality-specific priors. Finally, the Cross-Hierarchical Synergistic Network (CHSN) establishes bidirectional interactions between shared semantic and modality-aware features to mitigate quantization errors induced by weak alignment. Extensive experiments show that CrossWeaver achieves state-of-the-art performance on weakly aligned multimodal object detection benchmarks. Our method achieves the $\text{mAP}_{50}$ of 86.8% on DroneVehicle dataset.

## 🚀 Usage (Coming Soon)
Detailed instructions for environment setup, data preparation, training, and evaluation will be provided here once the code is public.

## 🔗 Citation
If you find our work or this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{yang2026crossweaver,
  title     = {CrossWeaver: Towards Efficient Cross-Modal Interweaving and Decoupling for Weakly-Aligned Multispectral Object Detection},
  author    = {Yang, Haitian and Fang, Juan and Zhu, Yiren and others},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year      = {2026}
}# CrossWeaver