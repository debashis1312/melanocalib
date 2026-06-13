import os
import gc
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from timm import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.data.mixup import Mixup

# Metric Libraries for Paper
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, classification_report, confusion_matrix
)

# ==============================
# 1. CONFIGURATION
# ==============================
MODEL_NAME = 'swin_large_patch4_window12_384.ms_in22k'

IMG_SIZE = 384
PHYSICAL_BATCH_SIZE = 8   
ACCUMULATION_STEPS = 4    # Effective Batch = 32

EPOCHS = 55
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.05
DROP_PATH_RATE = 0.3

# Mixup
MIXUP_ALPHA = 0.2
PROB_MIXUP = 0.5

# Paths
INPUT_DIR = "/kaggle/input/dermacon-in-dataset"
SAVE_DIR = "./checkpoints_paper"
os.makedirs(SAVE_DIR, exist_ok=True)

# Determinism
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# ==============================
# 2. DATA PREPARATION
# ==============================
print("Indexing images...")
image_path_map = {}
for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path_map[f] = os.path.join(root, f)

metadata_dir = None
for root, _, files in os.walk(INPUT_DIR):
    if "train_split.csv" in files:
        metadata_dir = root
        break
if not metadata_dir: metadata_dir = INPUT_DIR

train_df = pd.read_csv(os.path.join(metadata_dir, "train_split.csv"))
val_df = pd.read_csv(os.path.join(metadata_dir, "test_split.csv"))

TARGET_COL = "Main_Class" if "Main_Class" in train_df.columns else "Main_class"
ID_COL = "Image_name" if "Image_name" in train_df.columns else train_df.columns[0]
classes = sorted(train_df[TARGET_COL].dropna().unique().tolist())
label_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_label = {i: c for i, c in enumerate(classes)} # For reporting
num_classes = len(classes)
print(f"Classes ({num_classes}): {classes}")

# Weighted Sampler
class_counts = train_df[TARGET_COL].value_counts()
weights = 1.0 / class_counts.loc[train_df[TARGET_COL]].values
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# ==============================
# 3. TRANSFORMS
# ==============================
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

mixup_fn = Mixup(
    mixup_alpha=MIXUP_ALPHA, 
    cutmix_alpha=0.0, 
    prob=PROB_MIXUP, 
    switch_prob=0.0, 
    mode='batch',
    label_smoothing=0.1, 
    num_classes=num_classes
)

# ==============================
# 4. DATASETS
# ==============================
class DermaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = image_path_map.get(str(row[ID_COL])) or image_path_map.get(str(row[ID_COL]) + ".jpg")
        if p is None: return torch.zeros((3, IMG_SIZE, IMG_SIZE)), 0
        img = Image.open(p).convert("RGB")
        return (self.transform(img) if self.transform else img), label_to_idx[row[TARGET_COL]]

train_loader = DataLoader(DermaDataset(train_df, transform=train_transform), 
                          batch_size=PHYSICAL_BATCH_SIZE, sampler=sampler, 
                          num_workers=4, pin_memory=True, drop_last=True)

val_loader = DataLoader(DermaDataset(val_df, transform=val_transform), 
                        batch_size=PHYSICAL_BATCH_SIZE * 2, shuffle=False, 
                        num_workers=4, pin_memory=True)

# ==============================
# 5. MODEL
# ==============================
print(f"Loading Model: {MODEL_NAME}...")
model = create_model(MODEL_NAME, pretrained=True, num_classes=num_classes, 
                     drop_path_rate=DROP_PATH_RATE, use_checkpoint=True)

if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion_train = SoftTargetCrossEntropy()
criterion_val = nn.CrossEntropyLoss() # Added for tracking val loss
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS)
scaler = torch.cuda.amp.GradScaler()

# ==============================
# 6. TRAINING LOOP (WITH LOGGING)
# ==============================
def train():
    best_acc = 0.0
    history = []
    
    print(f"Starting Training: {EPOCHS} Epochs...")
    
    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [Train]")
        for i, (imgs, labs) in enumerate(pbar):
            imgs, labs = imgs.to(device), labs.to(device)
            imgs, mixed_labs = mixup_fn(imgs, labs)
            
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion_train(outputs, mixed_labs)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUMULATION_STEPS * imgs.size(0)
            _, max_mixed = mixed_labs.max(1)
            train_correct += (outputs.argmax(1) == max_mixed).sum().item()
            total_samples += imgs.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
        
        scheduler.step()
        epoch_train_acc = train_correct / total_samples
        epoch_train_loss = train_loss / total_samples
        
        # --- VALIDATE (4-WAY TTA + FULL METRICS) ---
        model.eval()
        val_loss_sum = 0
        all_preds, all_probs, all_tgts = [], [], []
        
        with torch.no_grad():
            for imgs, labs in tqdm(val_loader, desc="[Val + TTA]"):
                imgs, labs = imgs.to(device), labs.to(device)
                
                with torch.cuda.amp.autocast():
                    # 4-Way TTA
                    p1 = model(imgs)
                    p2 = model(torch.flip(imgs, [3]))
                    p3 = model(torch.flip(imgs, [2]))
                    p4 = model(torch.flip(imgs, [2, 3]))
                    logits = (p1 + p2 + p3 + p4) / 4.0
                    
                    # Track Val Loss (Important for papers)
                    v_loss = criterion_val(logits, labs)
                    val_loss_sum += v_loss.item() * imgs.size(0)
                
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_tgts.extend(labs.cpu().numpy())
        
        # Calculate Metrics
        val_acc = accuracy_score(all_tgts, all_preds)
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_f1 = f1_score(all_tgts, all_preds, average='weighted')
        val_prec = precision_score(all_tgts, all_preds, average='weighted', zero_division=0)
        val_rec = recall_score(all_tgts, all_preds, average='weighted', zero_division=0)
        
        try:
            val_auc = roc_auc_score(all_tgts, all_probs, multi_class='ovr')
        except:
            val_auc = 0.0 # Handle case where not all classes are present

        # Print Metrics clearly
        print(f"\n>> Epoch {epoch} Report:")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"   Val F1:     {val_f1:.4f} | Val AUC:   {val_auc:.4f}")
        print(f"   Precision:  {val_prec:.4f} | Recall:    {val_rec:.4f}")

        # Save History
        log_entry = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_auc": val_auc
        }
        history.append(log_entry)
        
        # Save JSON Logs (Overwrite every epoch)
        with open("training_log_paper.json", "w") as f:
            json.dump(history, f, indent=4)
        
        # Save Best Model & Detailed Report
        if val_acc > best_acc:
            best_acc = val_acc
            
            # Save Model
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                       os.path.join(SAVE_DIR, "best_model.pth"))
            
            # Save Classification Report (Per-Class Metrics)
            cls_report = classification_report(all_tgts, all_preds, target_names=classes, output_dict=True)
            with open("best_model_report.json", "w") as f:
                json.dump(cls_report, f, indent=4)
                
            print(f"🏆 New Best Model Saved! (Acc: {val_acc:.4f})")
            
            # Print Confusion Matrix for visual check
            print("\nConfusion Matrix (Best Model):")
            print(confusion_matrix(all_tgts, all_preds))

    print(f"\nFinal Best Accuracy: {best_acc:.4f}")
    print("Logs saved to 'training_log_paper.json' and 'best_model_report.json'")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    train()