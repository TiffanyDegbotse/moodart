"""
style_transfer.py
-----------------
Applies style transfer to a content image using one of three methods:
  naive  — Color LUT (per-channel statistics matching)
  kmeans — K-Means palette transfer
  neural — placeholder message (neural runs in the Flask app)

Usage:
    python scripts/style_transfer.py \
        --content  path/to/photo.jpg \
        --style    models/style_weights/vangogh.jpg \
        --method   kmeans \
        --strength 1.0 \
        --output   data/outputs/result.png
"""

import os
import argparse
import numpy as np
from PIL import Image


# ── Style transfer methods ────────────────────────────────────────────────────

def apply_naive_lut(content: Image.Image, style: Image.Image, strength: float = 1.0) -> Image.Image:
    """Per-channel mean/std statistics matching."""
    c = np.array(content).astype(np.float32)
    s = np.array(style).astype(np.float32)
    out = np.zeros_like(c)
    for i in range(3):
        cm, cs = c[:, :, i].mean(), c[:, :, i].std() + 1e-8
        sm, ss = s[:, :, i].mean(), s[:, :, i].std() + 1e-8
        transferred = (c[:, :, i] - cm) / cs * ss + sm
        out[:, :, i] = (1 - strength) * c[:, :, i] + strength * transferred
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_kmeans_palette(content: Image.Image, style: Image.Image,
                         n_colors: int = 16, strength: float = 1.0) -> Image.Image:
    """K-Means colour palette transfer."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    content_arr = np.array(content.convert('RGB')).astype(np.float32)
    style_arr   = np.array(style.convert('RGB')).astype(np.float32)
    h, w        = content_arr.shape[:2]
    c_pixels    = content_arr.reshape(-1, 3)
    s_pixels    = style_arr.reshape(-1, 3)

    c_sample = c_pixels[np.random.choice(len(c_pixels), min(5000, len(c_pixels)), replace=False)]
    s_sample = s_pixels[np.random.choice(len(s_pixels), min(5000, len(s_pixels)), replace=False)]

    c_kmeans  = KMeans(n_clusters=n_colors, random_state=42, n_init=5).fit(c_sample)
    s_kmeans  = KMeans(n_clusters=n_colors, random_state=42, n_init=5).fit(s_sample)
    c_palette = c_kmeans.cluster_centers_
    s_palette = s_kmeans.cluster_centers_

    dists   = pairwise_distances(c_palette, s_palette)
    mapping = dists.argmin(axis=1)

    pixel_dists = pairwise_distances(c_pixels, c_palette)
    assignments = pixel_dists.argmin(axis=1)

    transferred = np.zeros_like(c_pixels)
    for i in range(n_colors):
        mask = assignments == i
        if mask.any():
            style_color = s_palette[mapping[i]]
            transferred[mask] = (1 - strength) * c_pixels[mask] + strength * style_color

    transferred = np.clip(transferred, 0, 255).astype(np.uint8).reshape(h, w, 3)
    return Image.fromarray(transferred)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Apply style transfer to an image.')
    parser.add_argument('--content',  required=True, help='Path to content image')
    parser.add_argument('--style',    required=True, help='Path to style image')
    parser.add_argument('--method',   default='kmeans', choices=['naive', 'kmeans', 'neural'])
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--output',   default='data/outputs/stylized.png')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    content = Image.open(args.content).convert('RGB')
    style   = Image.open(args.style).convert('RGB')

    print(f'Content: {args.content}')
    print(f'Style:   {args.style}')
    print(f'Method:  {args.method} | Strength: {args.strength}')

    if args.method == 'naive':
        result = apply_naive_lut(content, style, strength=args.strength)
        print('Applied Color LUT style transfer.')
    elif args.method == 'kmeans':
        result = apply_kmeans_palette(content, style, strength=args.strength)
        print('Applied K-Means palette style transfer.')
    elif args.method == 'neural':
        print('Neural style transfer runs via the Flask app (app.py).')
        print('Pre-rendered outputs are in frontend/public/neural_gallery/')
        return

    result.save(args.output)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()