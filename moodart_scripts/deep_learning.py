"""
deep_learning.py
----------------
Trains VGG16 mood classifier with Optuna hyperparameter search.
Saves the best checkpoint to models/classifier/vgg_mood_best.pth.

Usage:
    python scripts/deep_learning.py \
        --data   data/processed \
        --output models/classifier \
        --epochs 15 \
        --trials 10
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from utils import fer_row_to_image, MOODS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset ───────────────────────────────────────────────────────────────────

class MoodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['mood'].values)
        self.classes = self.label_encoder.classes_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_arr = fer_row_to_image(self.df.loc[idx, 'pixels'])
        img = Image.fromarray(img_arr).convert('RGB').resize((224, 224))
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


# ── Model ─────────────────────────────────────────────────────────────────────

class MoodVGG(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.features   = vgg.features
        self.avgpool    = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 1024),         nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective_vgg(trial, train_loader, val_loader, criterion, num_classes):
    lr      = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    opt_name= trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    freeze  = trial.suggest_categorical('freeze_features', [True, False])

    model = MoodVGG(num_classes=num_classes, dropout=dropout).to(DEVICE)
    if freeze:
        for p in model.features.parameters():
            p.requires_grad = False

    if opt_name == 'Adam':
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, momentum=0.9)

    for _ in range(3):
        train_epoch(model, train_loader, criterion, opt)

    _, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
    return val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train VGG16 mood classifier.')
    parser.add_argument('--data',    default='data/processed')
    parser.add_argument('--output',  default='models/classifier')
    parser.add_argument('--epochs',  type=int, default=15)
    parser.add_argument('--trials',  type=int, default=10)
    parser.add_argument('--batch',   type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs('data/outputs', exist_ok=True)

    print(f'Device: {DEVICE}')

    df = pd.read_parquet(os.path.join(args.data, 'fer_train.parquet'))
    print(f'Loaded {len(df)} training samples')
    print(df['mood'].value_counts().to_string())

    # Class weights
    mood_order   = sorted(MOODS)
    class_counts = df['mood'].value_counts()
    weights = torch.tensor(
        [1.0 / class_counts[c] for c in mood_order], dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Train / val split
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train     = int(0.8 * len(df_shuffled))
    df_train    = df_shuffled[:n_train].reset_index(drop=True)
    df_val      = df_shuffled[n_train:].reset_index(drop=True)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = MoodDataset(df_train, transform=train_transform)
    val_dataset   = MoodDataset(df_val,   transform=val_transform)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch, shuffle=False,
                               num_workers=2, pin_memory=True)

    NUM_CLASSES = len(MOODS)
    print(f'\nTrain: {len(train_dataset)} | Val: {len(val_dataset)}')
    print(f'Classes: {train_dataset.classes}')

    # Optuna search
    print(f'\nRunning Optuna search ({args.trials} trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda t: objective_vgg(t, train_loader, val_loader, criterion, NUM_CLASSES),
        n_trials=args.trials, show_progress_bar=True
    )
    print(f'\nBest val accuracy: {study.best_value:.4f}')
    print(f'Best params: {study.best_params}')

    with open(os.path.join(args.output, 'vgg_best_params.pkl'), 'wb') as f:
        pickle.dump(study.best_params, f)

    # Optuna plots
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optuna VGGNet Optimization History')
    plt.tight_layout()
    plt.savefig('data/outputs/optuna_vgg_history.png', dpi=150)

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('VGGNet Hyperparameter Importances')
    plt.tight_layout()
    plt.savefig('data/outputs/optuna_vgg_param_importance.png', dpi=150)

    # Train final model
    best_lr      = study.best_params['lr']
    best_dropout = study.best_params['dropout']
    best_opt     = study.best_params['optimizer']
    best_freeze  = study.best_params['freeze_features']

    dl_model = MoodVGG(num_classes=NUM_CLASSES, dropout=best_dropout).to(DEVICE)
    if best_freeze:
        for p in dl_model.features.parameters():
            p.requires_grad = False

    if best_opt == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, dl_model.parameters()), lr=best_lr
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, dl_model.parameters()),
            lr=best_lr, momentum=0.9, weight_decay=1e-4
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc   = 0
    checkpoint_path = os.path.join(args.output, 'vgg_mood_best.pth')

    print(f'\nTraining final VGGNet for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(dl_model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_preds, vl_labels = eval_epoch(dl_model, val_loader, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(dl_model.state_dict(), checkpoint_path)
            print(f'  Best saved (val acc: {vl_acc:.4f})')

        print(f'Epoch {epoch+1}/{args.epochs} | Train: {tr_acc:.4f} | Val: {vl_acc:.4f}')

    print(f'\nBest val accuracy: {best_val_acc:.4f}')

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train'); ax1.plot(val_losses, label='Val')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(train_accs, label='Train'); ax2.plot(val_accs, label='Val')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.suptitle('VGGNet Training Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/outputs/training_curves.png', dpi=150)
    print('Saved data/outputs/training_curves.png')

    # Confusion matrix on val set
    dl_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    _, _, preds, labels = eval_epoch(dl_model, val_loader, criterion)
    print(classification_report(labels, preds, target_names=train_dataset.classes))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title('VGGNet Confusion Matrix (Validation)')
    plt.ylabel('True Mood'); plt.xlabel('Predicted Mood')
    plt.tight_layout()
    plt.savefig('data/outputs/vgg_confusion_matrix_validation.png', dpi=150)
    print('Saved data/outputs/vgg_confusion_matrix_validation.png')


if __name__ == '__main__':
    main()