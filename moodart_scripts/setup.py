"""
setup.py
--------
Creates project directories and downloads FER2013 from Kaggle.

Usage:
    python scripts/setup.py --kaggle        # download via Kaggle API
    python scripts/setup.py --no-kaggle     # skip download, create dirs only
"""

import os
import argparse
from pathlib import Path
from utils import make_dirs


def download_fer2013():
    """Download FER2013 dataset from Kaggle."""
    kaggle_json = Path('kaggle.json')
    if not kaggle_json.exists():
        raise FileNotFoundError(
            'kaggle.json not found. Download it from your Kaggle account settings '
            'and place it in the project root.'
        )
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    os.system('cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json')
    os.system('kaggle datasets download -d msambare/fer2013 -p data/raw --unzip')
    print('FER2013 downloaded to data/raw/')


def verify_download():
    """Print image counts per emotion per split."""
    for split in ['train', 'test']:
        split_path = Path('data/raw') / split
        if not split_path.exists():
            print(f'  {split}/ not found — run with --kaggle to download')
            continue
        print(f'\n{split.upper()} SET:')
        total = 0
        for emotion_dir in sorted(split_path.iterdir()):
            if emotion_dir.is_dir():
                count = len(list(emotion_dir.glob('*.jpg')) + list(emotion_dir.glob('*.png')))
                print(f'  {emotion_dir.name}: {count}')
                total += count
        print(f'  TOTAL: {total}')


def main():
    parser = argparse.ArgumentParser(description='MoodArt project setup.')
    parser.add_argument('--kaggle',    action='store_true', help='Download FER2013 via Kaggle API')
    parser.add_argument('--no-kaggle', action='store_true', help='Skip download, create dirs only')
    args = parser.parse_args()

    make_dirs()

    if args.kaggle:
        download_fer2013()
    else:
        print('Skipping download. Place FER2013 folders in data/raw/train/ and data/raw/test/')

    verify_download()


if __name__ == '__main__':
    main()