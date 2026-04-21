"""
evaluation.py
-------------
Evaluates all models on the official FER2013 test set.
Produces confusion matrices, a comparison bar chart, and GradCAM.

Usage:
    python scripts/evaluation.py \
        --data      data/processed \
        --models    models/classifier \
        --output    data/outputs
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
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

from utils import fer_row_to_image, MOODS
from features import extract_all_features
from naive_baseline import naive_mood_predictor
from deep_learning import MoodDataset, MoodVGG, eval_epoch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_test_features(df: pd.DataFrame, le: LabelEncoder):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting test features'):
        img = fer_row_to_image(row['pixels'])
        img_r = np.array(Image.fromarray(img).resize((64, 64)))
        X.append(extract_all_features(img_r))
        y.append(row['mood'])
    return np.array(X), le.transform(y)


def plot_cm(cm, classes, title, cmap, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Mood')
    plt.xlabel('Predicted Mood')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate all models on the test set.')
    parser.add_argument('--data',    default='data/processed')
    parser.add_argument('--models',  default='models/classifier')
    parser.add_argument('--output',  default='data/outputs')
    parser.add_argument('--batch',   type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load test parquet
    df_test = pd.read_parquet(os.path.join(args.data, 'fer_test.parquet'))
    print(f'Test set: {len(df_test)} samples')

    with open(os.path.join(args.data, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    results = {}

    # ── 1. Naive Baseline ─────────────────────────────────────────────────────
    naive_preds, naive_truths = [], []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc='Naive test'):
        img = fer_row_to_image(row['pixels'])
        img_resized = np.array(Image.fromarray(img).resize((64, 64)))
        naive_preds.append(naive_mood_predictor(img_resized))
        naive_truths.append(row['mood'])

    naive_acc = accuracy_score(naive_truths, naive_preds)
    results['Naive Baseline'] = naive_acc
    print(f'\nNaive Baseline Test Accuracy: {naive_acc:.4f}')

    cm = confusion_matrix(naive_truths, naive_preds, labels=sorted(MOODS))
    plot_cm(cm, sorted(MOODS), 'Naive Baseline Confusion Matrix (Test Set)',
            'Oranges', os.path.join(args.output, 'naive_confusion_matrix_test.png'))

    # ── 2. Classical ML ───────────────────────────────────────────────────────
    X_test, y_test = load_test_features(df_test, le)

    with open(os.path.join(args.models, 'rf_pipeline.pkl'), 'rb') as f:
        rf_pipeline = pickle.load(f)
    with open(os.path.join(args.models, 'svm_pipeline.pkl'), 'rb') as f:
        svm_pipeline = pickle.load(f)

    rf_test_acc  = accuracy_score(y_test, rf_pipeline.predict(X_test))
    svm_test_acc = accuracy_score(y_test, svm_pipeline.predict(X_test))
    results['Random Forest'] = rf_test_acc
    results['SVM']           = svm_test_acc
    print(f'\nRandom Forest Test Accuracy: {rf_test_acc:.4f}')
    print(f'SVM Test Accuracy: {svm_test_acc:.4f}')

    cm_rf = confusion_matrix(y_test, rf_pipeline.predict(X_test))
    plot_cm(cm_rf, le.classes_, 'Random Forest Confusion Matrix (Test Set)',
            'Blues', os.path.join(args.output, 'rf_confusion_matrix_test.png'))

    cm_svm = confusion_matrix(y_test, svm_pipeline.predict(X_test))
    plot_cm(cm_svm, le.classes_, 'SVM Confusion Matrix (Test Set)',
            'Greens', os.path.join(args.output, 'svm_confusion_matrix_test.png'))

    # ── 3. VGG16 ─────────────────────────────────────────────────────────────
    val_transform = __import__('torchvision').transforms.Compose([
        __import__('torchvision').transforms.ToTensor(),
        __import__('torchvision').transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = MoodDataset(df_test, transform=val_transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=2)

    # Load model
    checkpoint = os.path.join(args.models, 'vgg_mood_best.pth')
    if not os.path.exists(checkpoint):
        import requests
        print('Downloading VGG16 checkpoint from HuggingFace...')
        url = 'https://huggingface.co/tiffany101/modart_vgg/resolve/main/vgg_mood_best.pth'
        r = requests.get(url, stream=True)
        os.makedirs(args.models, exist_ok=True)
        with open(checkpoint, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Downloaded.')

    with open(os.path.join(args.models, 'vgg_best_params.pkl'), 'rb') as f:
        best_params = pickle.load(f)

    dl_model = MoodVGG(num_classes=len(MOODS), dropout=best_params['dropout']).to(DEVICE)
    dl_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    dl_model.eval()

    mood_order   = sorted(MOODS)
    class_counts = df_test['mood'].value_counts()
    weights = torch.tensor(
        [1.0 / class_counts.get(c, 1) for c in mood_order], dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    _, vgg_test_acc, vgg_preds, vgg_labels = eval_epoch(dl_model, test_loader, criterion)
    results['VGG16'] = vgg_test_acc
    print(f'\nVGG16 Test Accuracy: {vgg_test_acc:.4f}')
    print(classification_report(vgg_labels, vgg_preds, target_names=test_dataset.classes))

    cm_vgg = confusion_matrix(vgg_labels, vgg_preds)
    plot_cm(cm_vgg, test_dataset.classes, 'VGG16 Confusion Matrix (Test Set)',
            'Purples', os.path.join(args.output, 'vgg_confusion_matrix_test.png'))

    # ── Summary & bar chart ───────────────────────────────────────────────────
    print('\n=== Test Set Summary ===')
    for model_name, acc in results.items():
        print(f'{model_name}: {acc:.4f}')

    results_df = pd.DataFrame(results.items(), columns=['Model', 'Accuracy'])
    results_df.to_csv(os.path.join(args.output, 'model_comparison.csv'), index=False)

    colors = ['#ff9999', '#66b3ff', '#aaaaff', '#99ff99']
    plt.figure(figsize=(9, 5))
    bars = plt.bar(results_df['Model'], results_df['Accuracy'], color=colors, edgecolor='black')
    plt.ylim(0, 1.0)
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison: Mood Classification Accuracy (Test Set)')
    for bar, acc in zip(bars, results_df['Accuracy']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'model_comparison.png'), dpi=150)
    print(f"Saved {os.path.join(args.output, 'model_comparison.png')}")


if __name__ == '__main__':
    main()