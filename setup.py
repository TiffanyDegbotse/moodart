"""
setup.py — Download style images and verify model files.
Run once before starting the app: python setup.py
"""

import os
import io
import time
import requests
from PIL import Image

STYLE_IMAGES = {
    'monet': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_The_Water-Lily_Pond.jpg/1280px-Claude_Monet_-_The_Water-Lily_Pond.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/a/aa/Claude_Monet_-_The_Water-Lily_Pond.jpg',
    ],
    'vangogh': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/The_Starry_Night.jpg/1280px-The_Starry_Night.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/e/eb/The_Starry_Night.jpg',
    ],
    'kandinsky': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg/1280px-Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
    ],
    'hokusai': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1280px-Tsunami_by_hokusai_19th_century.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/a/a5/Tsunami_by_hokusai_19th_century.jpg',
    ],
    'munch': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/800px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
    ],
}

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

PLACEHOLDER_COLORS = {
    'monet':     (126, 184, 201),
    'vangogh':   (45,  74,  138),
    'kandinsky': (201, 75,  45),
    'hokusai':   (74,  124, 138),
    'munch':     (138, 106, 45),
}


def download_styles():
    os.makedirs('models/style_weights', exist_ok=True)
    for name, urls in STYLE_IMAGES.items():
        path = f'models/style_weights/{name}.jpg'
        if os.path.exists(path):
            print(f'  Already exists: {name}')
            continue
        success = False
        for i, url in enumerate(urls):
            try:
                time.sleep(1.5)
                r = requests.get(url, headers=HEADERS, timeout=30)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert('RGB')
                img.thumbnail((512, 512), Image.LANCZOS)
                img.save(path, format='JPEG', quality=92)
                print(f'  Downloaded: {name}')
                success = True
                break
            except Exception as e:
                if i < len(urls) - 1:
                    print(f'  URL {i+1} failed for {name}, trying next...')
                else:
                    print(f'  All URLs failed for {name}: {e}')
        if not success:
            color = PLACEHOLDER_COLORS.get(name, (128, 128, 128))
            Image.new('RGB', (512, 512), color).save(path, format='JPEG')
            print(f'  Created color placeholder for {name} — replace with real painting image')


def check_model():
    os.makedirs('models/classifier', exist_ok=True)
    path = 'models/classifier/vgg_mood_best.pth'
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f'  Classifier found: {path} ({size_mb:.1f} MB)')
    else:
        print(f'  No classifier found — using placeholder mood detection')
        print(f'  Copy vgg_mood_best.pth from Colab to: {path}')


if __name__ == '__main__':
    print('Setting up MoodArt...\n')
    print('Downloading style images:')
    download_styles()
    print('\nChecking model:')
    check_model()
    print('\nDone. Now run:')
    print('  cd backend && python app.py')
    print('  cd frontend && npm start')
