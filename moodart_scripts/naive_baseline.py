"""
naive_baseline.py
-----------------
Rule-based mood prediction from HSV color statistics.
No training required. Evaluated on the full dataset.

Usage:
    python scripts/naive_baseline.py \
        --input data/processed/fer_train.parquet \
        --output data/outputs
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import fer_row_to_image, MOODS
from features import extract_brightness_saturation


# ── Predictor ─────────────────────────────────────────────────────────────────

def naive_mood_predictor(img_rgb: np.ndarray) -> str:
    """
    Rule-based mood prediction from HSV color properties.
      High brightness + high saturation -> happy
      High brightness + low saturation  -> calm
      Low brightness + high contrast    -> dramatic
      High saturation + mid brightness  -> energetic
      Otherwise                         -> melancholic
    """
    bs = extract_brightness_saturation(img_rgb)
    hue, sat, val, brightness, contrast = bs

    if brightness > 0.6 and sat > 0.4:
        return 'happy'
    elif brightness > 0.6 and sat <= 0.4:
        return 'calm'
    elif brightness <= 0.4 and contrast > 0.25:
        return 'dramatic'
    elif sat > 0.5 and 0.35 < brightness <= 0.6:
        return 'energetic'
    else:
        return 'melancholic'


def evaluate_naive_baseline(df: pd.DataFrame):
    preds, truths = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Naive baseline'):
        img = fer_row_to_image(row['pixels'])
        img_resized = np.array(Image.fromarray(img).resize((64, 64)))
        preds.append(naive_mood_predictor(img_resized))
        truths.append(row['mood'])
    acc = accuracy_score(truths, preds)
    print(f'Naive Baseline Accuracy: {acc:.4f}')
    print(classification_report(truths, preds, target_names=sorted(MOODS)))
    return acc, preds, truths


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate naive baseline.')
    parser.add_argument('--input',  default='data/processed/fer_train.parquet')
    parser.add_argument('--output', default='data/outputs')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Loading {args.input}...')
    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} rows')

    acc, preds, truths = evaluate_naive_baseline(df)

    cm = confusion_matrix(truths, preds, labels=sorted(MOODS))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=sorted(MOODS), yticklabels=sorted(MOODS))
    plt.title('Naive Baseline Confusion Matrix')
    plt.ylabel('True Mood')
    plt.xlabel('Predicted Mood')
    plt.tight_layout()
    out_path = os.path.join(args.output, 'naive_confusion_matrix.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()