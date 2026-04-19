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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

# ── Model loading ─────────────────────────────────────────────────────────────
mood_classifier = None
style_model = None

def load_models():
    global mood_classifier, style_model

    try:
        import torch
        import torchvision.models as tv_models
        import torch.nn as nn

        class MoodVGG(nn.Module):
            def __init__(self, num_classes=5, dropout=0.5):
                super().__init__()
                vgg = tv_models.vgg16(pretrained=False)
                self.features = vgg.features
                self.avgpool = vgg.avgpool
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
                    nn.Linear(4096, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
                    nn.Linear(1024, num_classes)
                )
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return self.classifier(torch.flatten(x, 1))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = 'models/classifier/vgg_mood_best.pth'
        if os.path.exists(model_path):
            model = MoodVGG().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            mood_classifier = {'model': model, 'device': device}
            logger.info('Mood classifier loaded.')
        else:
            logger.warning('No mood classifier found — using placeholder.')
    except Exception as e:
        logger.warning(f'Could not load mood classifier: {e}')

    try:
        import tensorflow_hub as hub
        import tensorflow as tf

        @tf.autograph.experimental.do_not_convert
        def _load():
            return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

        style_model = _load()
        logger.info('Style transfer model loaded.')
    except Exception as e:
        logger.warning(f'Could not load style model: {e}')


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_image(data_url_or_b64):
    if ',' in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(',')[1]
    img_bytes = base64.b64decode(data_url_or_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    return img

def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def predict_mood_placeholder(img):
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

def predict_mood_vgg(img):
    import torch
    import torchvision.transforms as T
    CLASSES = ['calm', 'dramatic', 'energetic', 'happy', 'melancholic']
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(mood_classifier['device'])
    with torch.no_grad():
        probs = torch.softmax(mood_classifier['model'](tensor), dim=1).squeeze().cpu().numpy()
    pred_idx = probs.argmax()
    return {
        'mood': CLASSES[pred_idx],
        'confidence': round(float(probs[pred_idx]), 3),
        'scores': {c: round(float(p), 3) for c, p in zip(CLASSES, probs)}
    }

def apply_naive_lut(content, style, strength=1.0):
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

def apply_fast_nst(content, style, strength=1.0):
    import tensorflow as tf

    def to_tf(img, max_side=512):
        w, h = img.size
        scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
        if scale > 1.0:
            img = img.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        return tf.convert_to_tensor(arr)[tf.newaxis, ...]

    style_small = style.copy()
    style_small.thumbnail((256, 256), Image.LANCZOS)
    c_t = to_tf(content)
    s_t = to_tf(style_small, max_side=256)

    if strength < 1.0:
        gray = Image.new('RGB', style_small.size, (128, 128, 128))
        s_t = to_tf(gray, max_side=256) * (1.0 - strength) + s_t * strength

    out = style_model(c_t, s_t)[0]
    out = tf.squeeze(out, 0)
    out = tf.clip_by_value(out, 0.0, 1.0)
    return Image.fromarray((out.numpy() * 255).astype(np.uint8))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    import os
    files = os.listdir('/app') if os.path.exists('/app') else 'no /app'
    return jsonify({'status': 'ok', 'app_contents': files})

@app.route('/api/styles')
def get_styles():
    return jsonify(STYLES)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img = decode_image(data['image'])
        t0 = time.time()
        result = predict_mood_vgg(img) if mood_classifier else predict_mood_placeholder(img)
        mood = result['mood']

        return jsonify({
            'mood': mood,
            'confidence': result['confidence'],
            'scores': result['scores'],
            'recommended_style': MOOD_TO_STYLE[mood],
            'mood_description': MOOD_DESCRIPTIONS[mood],
            'style_info': STYLES[MOOD_TO_STYLE[mood]],
            'inference_time': round(time.time() - t0, 3),
            'model': 'VGGNet' if mood_classifier else 'Color Baseline (placeholder)'
        })
    except Exception as e:
        logger.error(f'Analyze error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/stylize', methods=['POST'])
def stylize():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img = decode_image(data['image'])
        style_key = data.get('style', 'vangogh')
        method = data.get('method', 'neural')
        strength = float(data.get('strength', 1.0))

        style_path = f'models/style_weights/{style_key}.jpg'

        logger.info(f'CWD: {os.getcwd()}')
        logger.info(f'Looking for: {style_path}')
        logger.info(f'Exists: {os.path.exists(style_path)}')
        logger.info(f'Weights folder: {os.listdir("models/style_weights") if os.path.exists("models/style_weights") else "NOT FOUND"}')

        if not os.path.exists(style_path):
            return jsonify({'error': f'Style {style_key} not found at {style_path}'}), 404

        style_img = Image.open(style_path).convert('RGB')
        t0 = time.time()

        if method == 'naive':
            result = apply_naive_lut(img, style_img, strength)
            model_name = 'Naive (Color LUT)'
        elif method == 'kmeans':
            result = apply_kmeans_palette(img, style_img, strength=strength)
            model_name = 'Classical ML (K-Means Palette)'
        elif style_model:
            result = apply_fast_nst(img, style_img, strength)
            model_name = 'Deep Learning (Fast NST)'
        else:
            result = apply_naive_lut(img, style_img, strength)
            model_name = 'Naive (Color LUT) — NST not loaded'

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
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
