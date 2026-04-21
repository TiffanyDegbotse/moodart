"""
data_prep.py
------------
Loads FER2013 from folder structure, applies mood mapping,
and saves processed dataframes to data/processed/.

Usage:
    python scripts/data_prep.py --root data/raw --output data/processed
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from utils import EMOTION_LABEL_TO_MOOD, MOODS


def load_fer2013_from_folders(root: str, split: str) -> pd.DataFrame:
    """
    Load FER2013 image folders and map emotions to mood labels.
    Images are converted to grayscale 48x48 and stored as pixel strings.
    disgust and fear are excluded — disgust has too few samples (436),
    fear is visually ambiguous and overlaps with surprise.
    """
    records = []
    split_path = Path(root) / split

    if not split_path.exists():
        raise FileNotFoundError(f'Split not found: {split_path}')

    for emotion_dir in sorted(split_path.iterdir()):
        if not emotion_dir.is_dir():
            continue
        emotion = emotion_dir.name.lower()
        mood = EMOTION_LABEL_TO_MOOD.get(emotion)
        if not mood:
            print(f'  Skipping: {emotion}')
            continue
        imgs = list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png'))
        for img_path in tqdm(imgs, desc=f'{emotion} -> {mood}', leave=False):
            img = Image.open(img_path).convert('L').resize((48, 48))
            pixels = ' '.join(map(str, np.array(img).flatten()))
            records.append({'pixels': pixels, 'mood': mood, 'path': str(img_path)})

    df = pd.DataFrame(records)
    print(f'\nLoaded {len(df)} images from {split}/')
    print(df['mood'].value_counts().to_string())
    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare FER2013 dataset.')
    parser.add_argument('--root',   default='data/raw')
    parser.add_argument('--output', default='data/processed')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print('=' * 50)
    print('Processing TRAIN split...')
    df_train = load_fer2013_from_folders(args.root, split='train')
    df_train.to_parquet(os.path.join(args.output, 'fer_train.parquet'), index=False)
    print(f'Saved fer_train.parquet ({len(df_train)} rows)')

    print('\n' + '=' * 50)
    print('Processing TEST split...')
    df_test = load_fer2013_from_folders(args.root, split='test')
    df_test.to_parquet(os.path.join(args.output, 'fer_test.parquet'), index=False)
    print(f'Saved fer_test.parquet ({len(df_test)} rows)')


if __name__ == '__main__':
    main()