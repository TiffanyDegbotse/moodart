"""
utils.py
--------
Shared constants, mood mapping, and image utility functions
used across all MoodArt scripts.
"""

import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

EMOTION_LABEL_TO_MOOD = {
    'happy':    'happy',
    'neutral':  'calm',
    'sad':      'melancholic',
    'surprise': 'energetic',
    'angry':    'dramatic',
}

MOOD_TO_STYLE = {
    'happy':       'monet',
    'calm':        'hokusai',
    'melancholic': 'vangogh',
    'energetic':   'kandinsky',
    'dramatic':    'munch',
}

MOODS = ['happy', 'calm', 'melancholic', 'energetic', 'dramatic']

DIRS = [
    'data/raw',
    'data/processed',
    'data/outputs',
    'models/classifier',
    'models/style_weights',
]


# ── Image helpers ─────────────────────────────────────────────────────────────

def fer_row_to_image(pixel_str: str, size: int = 48) -> np.ndarray:
    """Convert FER2013 pixel string to RGB numpy array."""
    pixels = np.array(pixel_str.split(), dtype=np.uint8).reshape(size, size)
    return np.stack([pixels, pixels, pixels], axis=-1)


def make_dirs():
    """Create all required project directories."""
    import os
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
    print('Directories ready.')