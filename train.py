#!/usr/bin/env python3
"""
Facial Recognition / Expression Classification - Modern Training Script (PyTorch)

Features
- Folder dataset (ImageFolder) with automatic or explicit train/val/test splits.
- Clean CLI (argparse) with sane defaults.
- Image augmentations & normalization.
- Choice of backbone: resnet18 (default) or efficientnet_b0 (pretrained optional).
- GPU/CPU auto-detection + AMP mixed precision.
- Optional backbone freezing for transfer learning.
- Cosine annealing LR schedule (or OneCycle).
- Early stopping, checkpointing (best + last), resume support.
- Deterministic seeding.
- Exports class mapping (class_to_idx.json) and training metrics (CSV).
- Optional TensorBoard logging.

Usage (example)
--------------
python train.py \
  --data_dir ./dataset \
  --output_dir ./outputs/exp1 \
  --epochs 25 --batch_size 64 --img_size 224 \
  --model resnet18 --pretrained --augment \
  --val_split 0.15 --test_split 0.1 \
  --amp --freeze_backbone 0

If your data is already split into subfolders:
dataset/
  train/
    classA/ ...
    classB/ ...
  val/
    classA/ ...
    classB/ ...
  test/
    classA/ ...
    classB/ ...

then call with: --data_dir ./dataset (splits will be inferred).

Author: Martin Badrous repo modernization
"""
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False

# -------------------------
# Utils
# -------------------------

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imagenet_norm():
    # ImageNet mean/std (most pretrained backbones expect this)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return mean, std

def build_transforms(img_size: int, augment: bool):
    mean, std = imagenet_norm()
    train_tf = [
        transforms.Resize((img_size, img_size)),
    ]
    if augment:
        train_tf = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transforms.Compose(train_tf), eval_tf

def stratified_split_indices(targets: List[int], val_split: float, test_split: float, seed: int = 42):
    """
    Create stratified indices for train/val/test from a flat ImageFolder targets list.
    """
    assert 0.0 <= val_split < 1.0 and 0.0 <= test_split < 1.0 and (val_split + test_split) < 1.0
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_split))
        n_val  = int(round(n * val_split))
        n_train = n - n_test - n_val
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test+n_val])
        train_idx.extend(idxs[n_test+n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def infer_splits(base: Path) -> Optional[Dict[str, Path]]:
    # Detect train/val/test subfolders if present
    candidates = {"train": base / "train", "val": base / "val", "test": base / "test"}
    if all(p.exists() and p.is_dir() for p in candidates.values()):
        return candidates
    # Accept train & val without test
    if (base / "train").exists() and (base / "val").exists():
        return {"train": base / "train", "val": base / "val", "test": None}
    return None

@dataclass
class Datasets:
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: Optional[torch.utils.data.Dataset]
    class_to_idx: Dict[str, int]

def build_datasets(data_dir: Path, img_size: int, augment: bool, val_split: float, test_split: float, seed: int) -> Datasets:
    split_paths = infer_splits(data_dir)
    train_tf, eval_tf = build_transforms(img_size, augment=augment)

    if split_paths:  # already split
        ds_train = datasets.ImageFolder(split_paths["train"], transform=train_tf)
        ds_val   = datasets.ImageFolder(split_paths["val"],   transform=eval_tf)
        ds_test  = datasets.ImageFolder(split_paths["test"],  transform=eval_tf) if split_paths.get("test") else None
        class_to_idx = ds_train.class_to_idx
        # ensure consistent mapping across splits
        if ds_val.class_to_idx != class_to_idx or (ds_test and ds_test.class_to_idx != class_to_idx):
            raise ValueError("class_to_idx mismatch across splits. Ensure identical class folders.")
        return Datasets(train=ds_train, val=ds_val, test=ds_test, class_to_idx=class_to_idx)

    # Single folder, do stratified split
    base_ds_no_tf = datasets.ImageFolder(data_dir)  # to access targets
    targets = [y for _, y in base_ds_no_tf.samples]
    train_idx, val_idx, test_idx = stratified_split_indices(targets, val_split, test_split, seed)

    # Recreate datasets with transforms using Subset
    base_ds_train = datasets.ImageFolder(data_dir, transform=train_tf)
    base_ds_eval  = datasets.ImageFolder(data_dir, transform=eval_tf)
    ds_train = Subset(base_ds_train, train_idx)
    ds_val   = Subset(base_ds_eval,  val_idx)
    ds_test  = Subset(base_ds_eval,  test_idx) if len(test_idx) > 0 else None
    return Datasets(train=ds_train, val=ds_val, test=ds_test, class_to_idx=base_ds_no_tf.class_to_idx)

def build_model(arch: str, num_classes: int, pretrained: bool, freeze_backbone: bool):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        backbone = [n for n, _ in model.named_parameters() if not n.startswith("fc")]
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
        backbone = [n for n, _ in model.named_parameters() if not n.startswith("classifier")]
    else:
        raise ValueError(f"Unsupported model: {arch}")
    if freeze_backbone:
        for name, p in model.named_parameters():
            if name in backbone:
                p.requires_grad = False
    return model

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def save_checkpoint(state: dict, is_best: bool, out_dir: Path, filename: str = "last.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / filename)
    if is_best:
        torch.save(state, out_dir / "best.pt")

def write_class_map(class_to_idx: Dict[str, int], out_dir: Path):
    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

def csv_logger_init(path: Path):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
    return f, writer

# -------------------------
# Train / Eval loops
# -------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits.detach(), targets) * bs
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, targets) * bs
        n += bs
    return running_loss / n, running_acc / n

# -------------------------
# Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Modern PyTorch training script for facial recognition/expression classification.")
    p.add_argument("--data_dir", type=str, required=True, help="Path to dataset root. Either one folder with class subfolders, or containing train/val(/test).")
    p.add_argument("--output_dir", type=str, default="./outputs/run", help="Directory to save checkpoints and logs.")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"], help="Backbone model.")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights.")
    p.add_argument("--freeze_backbone", type=int, default=0, help="Freeze backbone parameters (1) or not (0).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle", "none"])
    p.add_argument("--val_split", type=float, default=0.15, help="If data isn't split, fraction for validation.")
    p.add_argument("--test_split", type=float, default=0.10, help="If data isn't split, fraction for test.")
    p.add_argument("--augment", action="store_true", help="Enable data augmentation.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs without val acc improve).")
    p.add_argument("--save_every", type=int, default=0, help="Save checkpoint every N epochs (0 to disable).")
    p.add_argument("--log_tensorboard", action="store_true", help="Log to TensorBoard if available.")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & Loaders
    ds = build_datasets(Path(args.data_dir), args.img_size, args.augment, args.val_split, args.test_split, args.seed)
    num_classes = len(ds.class_to_idx)
    write_class_map(ds.class_to_idx, out_dir)

    train_loader = DataLoader(ds.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(ds.val,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(ds.test,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if ds.test else None

    # Model, Loss, Optim, Scheduler
    model = build_model(args.model, num_classes, pretrained=args.pretrained, freeze_backbone=bool(args.freeze_backbone))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle":
        steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume
    start_epoch, best_val_acc = 0, 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")

    # Logging
    csv_file, csv_writer = csv_logger_init(out_dir / "metrics.csv")
    writer = SummaryWriter(log_dir=str(out_dir / "tb")) if (args.log_tensorboard and TENSORBOARD_AVAILABLE) else None

    # Training loop
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler is not None and args.scheduler != "onecycle":
            scheduler.step()

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        csv_writer.writerow([epoch+1, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{current_lr:.6e}"])
        csv_file.flush()
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch+1)
            writer.add_scalar("Loss/val", val_loss, epoch+1)
            writer.add_scalar("Acc/train", train_acc, epoch+1)
            writer.add_scalar("Acc/val", val_acc, epoch+1)
            writer.add_scalar("LR", current_lr, epoch+1)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Checkpointing
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_val_acc": best_val_acc,
            "args": vars(args),
            "class_to_idx": ds.class_to_idx,
        }, is_best=is_best, out_dir=out_dir, filename="last.pt")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), out_dir / f"epoch_{epoch+1}.pth")

        # Early stopping
        if is_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

    csv_file.close()
    if writer:
        writer.close()

    # Final evaluation on test split if available
    if test_loader is not None:
        best_ckpt_path = out_dir / "best.pt"
        if best_ckpt_path.exists():
            state = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.4f}")
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"test_loss": test_loss, "test_acc": test_acc}, f, indent=2)

    # Export a convenient weights-only file
    torch.save(model.state_dict(), out_dir / "weights.pth")
    print(f"\nTraining complete. Artifacts saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
