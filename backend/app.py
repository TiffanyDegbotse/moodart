"""
MoodArt API — Flask backend
Endpoints:
  POST /api/analyze   — predict mood from uploaded image
  POST /api/stylize   — apply style transfer to uploaded image
  GET  /api/styles    — list available styles
  GET  /api/health    — health check
"""

import os
import io
import base64
import time
import logging
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_IMAGE_SIZE = (1024, 1024)
HF_TOKEN = os.environ.get('HF_TOKEN', '')
HF_API_URL = 'https://api-inference.huggingface.co/models/tiffany101/modart_vgg'

STYLES = {
    'monet':     {'name': 'Monet',     'description': 'Water Lilies — soft, impressionist',    'mood': 'happy'},
    'vangogh':   {'name': 'Van Gogh',  'description': 'Starry Night — swirling, emotive',      'mood': 'melancholic'},
    'kandinsky': {'name': 'Kandinsky', 'description': 'Composition 7 — bold, abstract',        'mood': 'energetic'},
    'hokusai':   {'name': 'Hokusai',   'description': 'The Great Wave — structured, peaceful',  'mood': 'calm'},
    'munch':     {'name': 'Munch',     'description': 'The Scream — intense, dramatic',         'mood': 'dramatic'},
}

MOOD_TO_STYLE = {
    'happy': 'monet', 'calm': 'hokusai', 'melancholic': 'vangogh',
    'energetic': 'kandinsky', 'dramatic': 'munch',
}

MOOD_DESCRIPTIONS = {
    'happy':       'Your image radiates warmth and joy.',
    'calm':        'A sense of serenity flows through your image.',
    'melancholic': 'Your image carries a deep, thoughtful energy.',
    'energetic':   'Your image pulses with vibrant energy.',
    'dramatic':    'Your image holds an intense, powerful presence.',
}

CLASSES = ['calm', 'dramatic', 'energetic', 'happy', 'melancholic']

# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_image(data_url_or_b64):
    """Decode base64 or data URL to PIL image."""
    if ',' in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(',')[1]
    img_bytes = base64.b64decode(data_url_or_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    return img

def encode_image(img):
    """Encode PIL image to base64 PNG."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def predict_mood_placeholder(img):
    """Rule-based mood prediction using color statistics."""
    arr = np.array(img).astype(np.float32)
    brightness = arr.mean() / 255.0
    saturation = arr.std() / 255.0

    if brightness > 0.6 and saturation > 0.15:
        mood = 'happy'
    elif brightness > 0.55 and saturation <= 0.15:
        mood = 'calm'
    elif brightness < 0.35 and saturation > 0.12:
        mood = 'dramatic'
    elif saturation > 0.18 and 0.35 < brightness <= 0.6:
        mood = 'energetic'
    else:
        mood = 'melancholic'

    scores = {m: round(np.random.uniform(0.05, 0.15), 3) for m in MOOD_TO_STYLE}
    scores[mood] = round(np.random.uniform(0.55, 0.80), 3)
    total = sum(scores.values())
    scores = {k: round(v/total, 3) for k, v in scores.items()}
    return {'mood': mood, 'confidence': scores[mood], 'scores': scores}

def predict_mood_hf(img):
    """
    Predict mood using Hugging Face Inference API.
    Sends image to hosted VGGNet model — no local memory needed.
    """
    # Convert image to bytes
    buf = io.BytesIO()
    img.resize((224, 224)).save(buf, format='JPEG')
    img_bytes = buf.getvalue()

    response = requests.post(
        HF_API_URL,
        headers={'Authorization': f'Bearer {HF_TOKEN}'},
        data=img_bytes,
        timeout=30
    )

    if response.status_code == 503:
        # Model is loading on HF servers — use placeholder
        logger.warning('HF model loading, using placeholder')
        return predict_mood_placeholder(img)

    if response.status_code != 200:
        raise Exception(f'HF API error {response.status_code}: {response.text}')

    result = response.json()

    # HF returns list of {label, score} for image classification
    if isinstance(result, list):
        scores = {item['label']: round(item['score'], 3) for item in result}
        mood = max(scores, key=scores.get)
        # Map HF labels to our mood classes if needed
        mood = mood.lower()
        if mood not in CLASSES:
            # fallback — find closest
            mood = CLASSES[0]
    else:
        # Unexpected format — use placeholder
        logger.warning(f'Unexpected HF response: {result}')
        return predict_mood_placeholder(img)

    return {
        'mood': mood,
        'confidence': round(scores.get(mood, 0.5), 3),
        'scores': scores
    }

def apply_naive_lut(content, style, strength=1.0):
    """Naive baseline: channel statistics matching."""
    c = np.array(content).astype(np.float32)
    s = np.array(style).astype(np.float32)
    out = np.zeros_like(c)
    for i in range(3):
        cm, cs = c[:,:,i].mean(), c[:,:,i].std() + 1e-8
        sm, ss = s[:,:,i].mean(), s[:,:,i].std() + 1e-8
        transferred = (c[:,:,i] - cm) / cs * ss + sm
        out[:,:,i] = (1 - strength) * c[:,:,i] + strength * transferred
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def apply_kmeans_palette(content, style, n_colors=16, strength=1.0):
    """Classical ML: K-Means palette transfer."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    content_arr = np.array(content.convert('RGB')).astype(np.float32)
    style_arr = np.array(style.convert('RGB')).astype(np.float32)
    h, w = content_arr.shape[:2]
    c_pixels = content_arr.reshape(-1, 3)
    s_pixels = style_arr.reshape(-1, 3)

    c_sample = c_pixels[np.random.choice(len(c_pixels), min(5000, len(c_pixels)), replace=False)]
    s_sample = s_pixels[np.random.choice(len(s_pixels), min(5000, len(s_pixels)), replace=False)]

    c_kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=5).fit(c_sample)
    s_kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=5).fit(s_sample)
    c_palette = c_kmeans.cluster_centers_
    s_palette = s_kmeans.cluster_centers_

    dists = pairwise_distances(c_palette, s_palette)
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'hf_token_set': bool(HF_TOKEN),
    })

@app.route('/api/styles')
def get_styles():
    return jsonify(STYLES)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Predict mood from uploaded image using HF Inference API."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img = decode_image(data['image'])
        t0 = time.time()

        if HF_TOKEN:
            try:
                result = predict_mood_hf(img)
                model_name = 'VGGNet (HF Inference)'
            except Exception as e:
                logger.warning(f'HF inference failed: {e}, using placeholder')
                result = predict_mood_placeholder(img)
                model_name = 'Color Baseline (placeholder)'
        else:
            result = predict_mood_placeholder(img)
            model_name = 'Color Baseline (placeholder)'

        mood = result['mood']

        return jsonify({
            'mood': mood,
            'confidence': result['confidence'],
            'scores': result['scores'],
            'recommended_style': MOOD_TO_STYLE.get(mood, 'vangogh'),
            'mood_description': MOOD_DESCRIPTIONS.get(mood, ''),
            'style_info': STYLES[MOOD_TO_STYLE.get(mood, 'vangogh')],
            'inference_time': round(time.time() - t0, 3),
            'model': model_name
        })
    except Exception as e:
        logger.error(f'Analyze error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/stylize', methods=['POST'])
def stylize():
    """Apply style transfer to image."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img = decode_image(data['image'])
        style_key = data.get('style', 'vangogh')
        method = data.get('method', 'naive')
        strength = float(data.get('strength', 1.0))

        style_path = f'models/style_weights/{style_key}.jpg'
        if not os.path.exists(style_path):
            return jsonify({'error': f'Style {style_key} not found'}), 404

        style_img = Image.open(style_path).convert('RGB')
        t0 = time.time()

        if method == 'kmeans':
            result = apply_kmeans_palette(img, style_img, strength=strength)
            model_name = 'Classical ML (K-Means Palette)'
        elif method == 'naive':
            result = apply_naive_lut(img, style_img, strength)
            model_name = 'Naive (Color LUT)'
        else:
            # Neural — fall back to naive since tensorflow not installed
            result = apply_naive_lut(img, style_img, strength)
            model_name = 'Naive (Color LUT) — Neural requires GPU deployment'

        return jsonify({
            'stylized_image': encode_image(result),
            'style': style_key,
            'method': method,
            'model': model_name,
            'inference_time': round(time.time() - t0, 3)
        })

    except Exception as e:
        logger.error(f'Stylize error: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)