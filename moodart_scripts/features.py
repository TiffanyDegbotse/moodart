"""
features.py
-----------
Extracts color histogram + HOG + HSV features from FER2013 images
and saves X / y arrays for classical ML training.

Usage:
    python scripts/features.py \
        --input  data/processed/fer_train.parquet \
        --output data/processed \
        --max-samples 5000
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage.color import rgb2gray

from utils import fer_row_to_image


# ── Feature functions ─────────────────────────────────────────────────────────

def extract_color_histogram(img_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    features = []
    for ch in range(3):
        hist, _ = np.histogram(img_rgb[:, :, ch], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.append(hist)
    return np.concatenate(features)


def extract_hog_features(img_rgb: np.ndarray) -> np.ndarray:
    gray = rgb2gray(img_rgb)
    return hog(
        gray, orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    ).astype(np.float32)


def extract_brightness_saturation(img_rgb: np.ndarray) -> np.ndarray:
    img_hsv = np.array(Image.fromarray(img_rgb).convert('HSV'))
    mean_hue        = img_hsv[:, :, 0].mean() / 255.0
    mean_sat        = img_hsv[:, :, 1].mean() / 255.0
    mean_val        = img_hsv[:, :, 2].mean() / 255.0
    mean_brightness = img_rgb.mean() / 255.0
    contrast        = img_rgb.std() / 255.0
    return np.array([mean_hue, mean_sat, mean_val, mean_brightness, contrast], dtype=np.float32)


def extract_all_features(img_rgb: np.ndarray) -> np.ndarray:
    """Combine color histogram + HOG + HSV into a single feature vector (1669-dim)."""
    return np.concatenate([
        extract_color_histogram(img_rgb),
        extract_hog_features(img_rgb),
        extract_brightness_saturation(img_rgb),
    ])


def build_feature_matrix(df: pd.DataFrame, max_samples: int):
    df_sample = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    X, y = [], []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc='Extracting features'):
        img = fer_row_to_image(row['pixels'])
        img_resized = np.array(Image.fromarray(img).resize((64, 64)))
        X.append(extract_all_features(img_resized))
        y.append(row['mood'])
    return np.array(X), np.array(y)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract features from FER2013.')
    parser.add_argument('--input',       default='data/processed/fer_train.parquet')
    parser.add_argument('--output',      default='data/processed')
    parser.add_argument('--max-samples', type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Loading {args.input}...')
    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} rows')
    print(df['mood'].value_counts().to_string())

    print(f'\nExtracting features (max {args.max_samples} samples)...')
    X, y = build_feature_matrix(df, max_samples=args.max_samples)
    print(f'Feature matrix: {X.shape}')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f'Classes: {le.classes_}')

    np.save(os.path.join(args.output, 'X.npy'), X)
    np.save(os.path.join(args.output, 'y.npy'), y_encoded)
    with open(os.path.join(args.output, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    print(f'\nSaved X.npy, y.npy, label_encoder.pkl to {args.output}')


if __name__ == '__main__':
    main()