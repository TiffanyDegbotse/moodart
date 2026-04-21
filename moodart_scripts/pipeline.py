"""
pipeline.py
-----------
End-to-end pipeline test: loads a photo, predicts mood, applies style transfer.

Usage:
    python scripts/pipeline.py \
        --image    path/to/photo.jpg \
        --models   models/classifier \
        --styles   models/style_weights \
        --method   kmeans \
        --output   data/outputs
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from utils import MOOD_TO_STYLE, MOODS, fer_row_to_image
from style_transfer import apply_naive_lut, apply_kmeans_palette
from deep_learning import MoodVGG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES   = ['calm', 'dramatic', 'energetic', 'happy', 'melancholic']
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_mood(img: Image.Image, model: MoodVGG) -> dict:
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    pred_idx = probs.argmax()
    return {
        'mood':       CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'scores':     {c: float(p) for c, p in zip(CLASSES, probs)},
    }


def main():
    parser = argparse.ArgumentParser(description='Run the full MoodArt pipeline.')
    parser.add_argument('--image',   required=True)
    parser.add_argument('--models',  default='models/classifier')
    parser.add_argument('--styles',  default='models/style_weights')
    parser.add_argument('--method',  default='kmeans', choices=['naive', 'kmeans'])
    parser.add_argument('--strength',type=float, default=1.0)
    parser.add_argument('--output',  default='data/outputs')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load model
    checkpoint = os.path.join(args.models, 'vgg_mood_best.pth')
    params_path = os.path.join(args.models, 'vgg_best_params.pkl')

    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)
        dropout = best_params.get('dropout', 0.694)
    else:
        dropout = 0.694  # fallback to known best value

    model = MoodVGG(num_classes=len(MOODS), dropout=dropout).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    print(f'Model loaded from {checkpoint}')

    # Load and predict
    img = Image.open(args.image).convert('RGB')
    result = predict_mood(img, model)

    mood            = result['mood']
    confidence      = result['confidence']
    style_name      = MOOD_TO_STYLE[mood]
    style_path      = os.path.join(args.styles, f'{style_name}.jpg')

    print(f'\nImage:      {args.image}')
    print(f'Mood:       {mood} ({confidence:.1%} confidence)')
    print(f'Style:      {style_name}')
    print('\nAll scores:')
    for m, s in sorted(result['scores'].items(), key=lambda x: -x[1]):
        print(f'  {m}: {s:.3f}')

    if not os.path.exists(style_path):
        print(f'\nStyle image not found: {style_path}')
        print('Place style .jpg files in models/style_weights/ to apply style transfer.')
        return

    style_img = Image.open(style_path).convert('RGB')

    if args.method == 'naive':
        stylized = apply_naive_lut(img, style_img, strength=args.strength)
    else:
        stylized = apply_kmeans_palette(img, style_img, strength=args.strength)

    out_path = os.path.join(args.output, f'pipeline_{mood}_{style_name}.png')
    stylized.save(out_path)
    print(f'\nStylized image saved: {out_path}')


if __name__ == '__main__':
    main()