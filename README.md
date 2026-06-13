# MelanoCalib 🩺

**Skin Tone-Aware Deep Learning for Multi-Class Skin Disorder Classification**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-71.82%25-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-DermaCon--IN-purple)](https://arxiv.org/abs/2506.06099)

> B.Tech Final Year Project — Department of Information Technology, JIS College of Engineering, Kalyani  
> Guided by **Dr. Sumit Das**, Assistant Professor, Dept. of IT

---

## 📌 Overview

MelanoCalib is a deep learning framework for automated multi-class skin disorder classification from clinical images. It addresses a critical gap in existing dermatological AI systems — their poor performance on **darker skin tones (Fitzpatrick types IV–VI)**, which are underrepresented in most public datasets.

The framework is trained and evaluated on the **DermaCon-IN** dataset — a clinically annotated Indian skin disorder dataset covering eight disease categories, with meaningful representation of South Asian skin tones.

---

## 🎯 Key Features

- **Skin tone-aware** — trained on DermaCon-IN, which includes Fitzpatrick types IV–VI from Indian outpatient clinics
- **Swin Transformer (Large)** backbone with 197M parameters for hierarchical feature learning
- **4-way Test Time Augmentation (TTA)** for robust inference
- **Mixup augmentation** with Soft Target Cross-Entropy for smoother generalization
- **Weighted Random Sampling** to handle long-tailed class distributions
- **Comprehensive metrics** — Accuracy, Weighted F1, Precision, Recall, and AUC per epoch

---

## 🏥 Dataset — DermaCon-IN

| Property | Details |
|---|---|
| Source | Indian outpatient dermatology clinics |
| Skin types | Fitzpatrick IV–VI (predominantly) |
| Total classes | 8 |
| Train split | ~4,400 images |
| Image type | Clinical photographs (non-dermoscopic) |

**Disease categories:**

| # | Category |
|---|---|
| 1 | Infectious Disorders |
| 2 | Inflammatory Disorders |
| 3 | Pigmentary Disorders |
| 4 | Skin Appendage Disorders |
| 5 | Neoplasms |
| 6 | Keratinisation Disorders |
| 7 | Eczema and Dermatitis |
| 8 | No Definite Diagnosis |

Dataset paper: [DermaCon-IN — arXiv:2506.06099](https://arxiv.org/abs/2506.06099)

---

## 📊 Results

### Overall Performance

| Model | Pre-training | Accuracy | Weighted F1 | Precision |
|---|---|---|---|---|
| ResNet50 (Baseline) | ImageNet-1k | 47.45% | 0.4643 | 46.59% |
| EfficientNet-B4 (Baseline) | ImageNet-1k | 64.28% | 0.6520 | 64.62% |
| Swin-Base (Paper Baseline) | ImageNet-22k | 70.41% | 0.6969 | 69.83% |
| **Swin-Large (MelanoCalib)** | **ImageNet-22k** | **71.82%** | **0.7165** | **71.50%** |

### Class-wise F1 Highlights

| Class | F1-Score |
|---|---|
| Skin Appendage Disorders | **0.7912** |
| Pigmentary Disorders | **0.7845** |
| Inflammatory Disorders | 0.6390 |
| Neoplasms | 0.4480 |
| Keratinisation Disorders | 0.2340 |

> Strong performance on pigmentary disorders (+vs. fair-skin-trained models) provides empirical evidence that inclusive training data directly improves feature extraction quality for darker skin tones.

---

## 🗂️ Repository Structure

```
melanocalib/
├── train.py               # Main training script
├── train_split.csv        # Training data split
├── test_split.csv         # Validation/test data split
├── Skin_Metadata.csv      # Image metadata
└── README.md
```

---

## ⚙️ Setup and Installation

### Prerequisites

```bash
Python >= 3.9
CUDA >= 11.7 (GPU recommended)
```

### Install Dependencies

```bash
pip install torch torchvision timm
pip install pandas numpy pillow tqdm scikit-learn
```

---

## 🚀 Training

### 1. Add the Dataset (Kaggle)

This script is designed to run as a **Kaggle Notebook**. Add the DermaCon-IN dataset from the Kaggle dataset hub. It will be mounted automatically at:

```
/kaggle/input/dermacon-in-dataset/
```

The script expects `train_split.csv` and `test_split.csv` inside that directory, along with all image files (`.jpg` / `.png`).

### 2. Run Training

```bash
python train.py
```

### 3. Key Hyperparameters

| Parameter | Value |
|---|---|
| Model | `swin_large_patch4_window12_384` |
| Image size | 384 × 384 |
| Effective batch size | 32 (8 physical × 4 accumulation steps) |
| Epochs | 55 |
| Learning rate | 2e-5 (AdamW) |
| Weight decay | 0.05 |
| Drop path rate | 0.3 |
| Mixup alpha | 0.2 |
| TTA | 4-way (H-flip, V-flip, HV-flip) |

### 4. Outputs

| File | Description |
|---|---|
| `checkpoints_paper/best_model.pth` | Best model weights (by val accuracy) |
| `training_log_paper.json` | Per-epoch metrics log |
| `best_model_report.json` | Per-class precision/recall/F1 at best epoch |

---

## 🔬 Architecture

MelanoCalib uses the **Swin Transformer (Large)** — a hierarchical vision transformer that computes self-attention within local, non-overlapping windows and shifts them across layers to capture cross-window context.

```
Clinical Image (384×384)
       ↓
  Patch Partition + Linear Embedding
       ↓
  Swin Transformer Stage 1  →  96-dim features
       ↓
  Swin Transformer Stage 2  →  192-dim features
       ↓
  Swin Transformer Stage 3  →  384-dim features
       ↓
  Swin Transformer Stage 4  →  768-dim features
       ↓
  Global Average Pooling
       ↓
  Linear Classification Head (8 classes)
       ↓
  Softmax → Class Probabilities
```

At inference, **4-way TTA** (original + 3 flipped variants) is averaged to produce the final prediction.

---

## 📈 Training Strategy

- **Weighted Random Sampler** — oversamples minority classes to counteract the long-tailed DermaCon-IN distribution
- **Mixup (α=0.2)** with **Soft Target Cross-Entropy** — smooths decision boundaries, reduces overconfidence
- **RandAugment + Random Erasing** — simulates real-world clinical image variability
- **Gradient Accumulation** (4 steps) — achieves effective batch size of 32 on limited VRAM
- **Mixed Precision (AMP)** — speeds up training with `torch.cuda.amp`
- **Cosine Annealing Warm Restarts** — periodically resets LR to escape local minima

---

## 🔮 Future Work

- [ ] Enable CutMix alongside Mixup for stronger regularization
- [ ] Layer-wise LR decay — lower rates for backbone, higher for head
- [ ] EMA (Exponential Moving Average) weights for smoother validation
- [ ] 8-way TTA with rotations (90°, 180°, 270°)
- [ ] Multimodal fusion — incorporate patient metadata (age, skin type, symptom duration)
- [ ] Self-supervised pretraining on unlabeled dermatology images
- [ ] Mobile deployment via knowledge distillation and quantization
- [ ] Federated learning for multi-hospital training without data sharing

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{melanocalib2026,
  title   = {MelanoCalib: Skin Tone-Aware Deep Learning for Multi-Class Skin Disorder Classification},
  author  = {Debashis Paul},
  year    = {2026},
  school  = {JIS College of Engineering, Kalyani},
  note    = {B.Tech Final Year Project, Department of Information Technology}
}
```

For the DermaCon-IN dataset:

```bibtex
@article{madarkar2026dermaconin,
  title   = {DermaCon-IN: A multi-concept annotated dermatological image dataset of Indian skin disorders for clinical AI research},
  author  = {Madarkar, S. S. and Madarkar, M. and Venkatesh, M. et al.},
  journal = {arXiv preprint arXiv:2506.06099},
  year    = {2026}
}
```

---

## 🙏 Acknowledgements

- **Dr. Sumit Das**, Department of IT, JIS College of Engineering — project guidance
- The board-certified dermatologists and clinical staff at the collaborating hospitals who annotated the DermaCon-IN dataset
- The [timm](https://github.com/huggingface/pytorch-image-models) library for the Swin Transformer implementation
- The open-source research community for publicly available tools and architectures

---

## 📬 Contact

**Debashis Paul**  
B.Tech Information Technology, JIS College of Engineering  
University Roll No.: 123221104027  
GitHub: [@debashis1312](https://github.com/debashis1312)
