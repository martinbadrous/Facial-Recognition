# 👁️ Facial Recognition / Expression Classification

A **modern PyTorch-based pipeline** for training and evaluating facial expression or identity recognition models.  
Developed by **[Martin Badrous](https://github.com/martinbadrous)**.

---

## 🌍 Overview

This repository provides a **clean, modular, and high-performance** deep learning training framework for facial recognition and expression classification.  
It supports **custom datasets**, **transfer learning**, **data augmentation**, and **GPU acceleration** — ideal for research, prototyping, and production.

The project is a complete refactor of the earlier notebook implementation, now rewritten as a professional, reusable **command-line training script**.

---

## ⚙️ Features

✅ Support for **custom datasets** using folder structure  
✅ Built with **PyTorch** and **Torchvision**  
✅ Transfer learning via **ResNet18** or **EfficientNet-B0**  
✅ **Automatic dataset splitting** (train/val/test)  
✅ Mixed precision (**AMP**) for faster training  
✅ **Early stopping**, **checkpointing**, and **resume** support  
✅ CSV + TensorBoard logging  
✅ Class-to-index mapping & metrics export  
✅ Lightweight and easily customizable  

---

## 🗂️ Repository Structure

```bash
Facial-Recognition/
├── train.py                    # Main training script (modernized)
├── requirements.txt             # Python dependencies
├── dataset/                     # Your dataset folder (see below)
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   └── neutral/
└── outputs/
    └── exp1/
        ├── best.pt
        ├── last.pt
        ├── weights.pth
        ├── class_to_idx.json
        ├── metrics.csv
        └── test_metrics.json
```

---

## 🧠 Dataset Format

You can use either of the following formats:

### **Single folder (auto-split)**
```
dataset/
├── happy/
├── sad/
├── angry/
└── neutral/
```

The script automatically splits the data into training, validation, and test sets (configurable via `--val_split` and `--test_split`).

### **Pre-split dataset**
```
dataset/
├── train/
│   ├── happy/ ...
│   ├── sad/ ...
│   └── angry/ ...
├── val/
│   ├── happy/ ...
│   └── sad/ ...
└── test/
    ├── happy/ ...
    └── sad/ ...
```
The script automatically detects these subfolders and uses them directly.

---

## 🚀 Quick Setup & Training

💡 You can copy and paste all commands below at once.

```bash
# 1. Clone the repository
git clone https://github.com/martinbadrous/Facial-Recognition.git
cd Facial-Recognition

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train a model (example)
python train.py   --data_dir ./dataset   --output_dir ./outputs/exp1   --epochs 25 --batch_size 64 --img_size 224   --model resnet18 --pretrained --augment   --val_split 0.15 --test_split 0.10   --amp

# 5. Optional: enable TensorBoard logging
python train.py ... --log_tensorboard
```

---

## 🧩 Model Options

| Option | Description |
|--------|--------------|
| `--model` | `resnet18` (default) or `efficientnet_b0` |
| `--pretrained` | Use ImageNet pretrained weights |
| `--freeze_backbone` | Freeze feature extractor for transfer learning |
| `--augment` | Enable data augmentation (flip, color jitter, etc.) |
| `--amp` | Use automatic mixed precision |
| `--scheduler` | `cosine`, `onecycle`, or `none` |
| `--val_split`, `--test_split` | Custom dataset split ratios |

---

## 📊 Output Files

All training artifacts are saved under the specified `--output_dir` (default: `./outputs/run`):

| File | Description |
|------|--------------|
| `best.pt` | Best model checkpoint (highest val accuracy) |
| `last.pt` | Last epoch checkpoint |
| `weights.pth` | Weights-only file for inference |
| `metrics.csv` | Per-epoch training/validation metrics |
| `test_metrics.json` | Final test results |
| `class_to_idx.json` | Mapping of class names to indices |

---

## 💡 Example Output

**Training log:**
```
Epoch 10/25
  train_loss=0.2451  train_acc=0.9102  |  val_loss=0.3321  val_acc=0.8850
```

**metrics.csv**
| epoch | train_loss | train_acc | val_loss | val_acc | lr |
|--------|-------------|-----------|-----------|----------|------|
| 1 | 1.21 | 0.57 | 1.03 | 0.61 | 3.00e-04 |
| 2 | 0.82 | 0.73 | 0.65 | 0.77 | 2.95e-04 |
| ... | ... | ... | ... | ... | ... |

**Final test result:**
```json
{
  "test_loss": 0.3021,
  "test_acc": 0.8923
}
```

---

## 🧪 Evaluate on Test Set

After training completes, the script automatically evaluates the best checkpoint (`best.pt`) on the test set (if available) and saves the results to:
```
outputs/exp1/test_metrics.json
```

---

## 🧰 Requirements

```bash
torch>=2.1.0
torchvision>=0.16.0
tensorboard>=2.12.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧭 Roadmap

- [ ] Add `infer.py` for real-time predictions (image/video/webcam)
- [ ] Add ONNX / TorchScript export for deployment
- [ ] Add Grad-CAM visualizations
- [ ] Support for larger EfficientNet variants
- [ ] Integrate with a simple web demo using Streamlit or Gradio

---

## 👨‍💻 Author

**Martin Badrous**  
Computer Vision & Robotics Engineer  
🎓 M.Sc. in Computer Vision and Robotics  
📍 Based in France | 🇪🇬 Egyptian origin  
🔗 [GitHub Profile](https://github.com/martinbadrous)

---

⭐ **If this project helps your research or development, please give it a star on GitHub!**
