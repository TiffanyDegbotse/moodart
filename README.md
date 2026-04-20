# MoodArt (AI-Powered Mood-Based Style Transfer)

> AIPI 540 Final Project — Upload a photo, detect its emotional mood, transform it into art.

## Live Demo
- **Frontend**: Deployed on Railway
- **Backend API**: Deployed on Railway

---

## How It Works

1. User uploads a photo
2. VGGNet mood classifier detects the emotional mood (happy / calm / melancholic / energetic / dramatic)
3. System recommends an artistic style matching the mood
4. Neural style transfer transforms the image into the chosen artwork style

---

## Three ML Approaches

| Model | Approach | Purpose |
|-------|----------|---------|
| Naive Baseline | Color LUT (channel statistics) | Fast rule-based color matching |
| Classical ML | K-Means palette transfer | Unsupervised color clustering |
| Deep Learning | VGGNet + Fast Neural Style Transfer | Full artistic transformation |

---

## Project Structure

```
moodart_app/
└── moodart/
├── backend/
│ ├── app.py ← Flask API
│ ├── requirements.txt
│ ├── railway.toml
│ └── models/
│ ├── classifier/
│ │ └── vgg_mood_best.pth ← trained VGG16 mood classifier
│ └── style_weights/ ← downloaded style images
├── frontend/
│ ├── package.json
│ ├── public/index.html
│ └── src/
│ ├── App.js
│ ├── App.css
│ ├── index.js
│ └── index.css
├── moodart_scripts/ ← ML pipeline (training + experiments)
│ ├── data_prep.py
│ ├── features.py
│ ├── naive_baseline.py
│ ├── classical_ml.py
│ ├── deep_learning.py
│ ├── evaluation.py
│ ├── style_transfer.py
│ └── pipeline.py
├── data/
│ ├── raw/ ← FER2013 dataset
│ ├── processed/
│ └── outputs/
├── notebooks/ ← exploration notebooks
├── setup.py
├── README.md
└── .gitignore
```

---

## Setup & Running Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:5000 npm start
```

---

## Deploying to Railway

### Backend service
1. Create new Railway project
2. Add service → GitHub repo → select `backend/` as root
3. Set env vars: none required for basic deployment
4. Railway auto-detects `railway.toml` and deploys

### Frontend service
1. Add another service in same project
2. Select `frontend/` as root
3. Set env var: `REACT_APP_API_URL=https://your-backend-url.railway.app`
4. Build command: `npm run build`
5. Start command: `npx serve -s build`

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/styles` | GET | List available styles |
| `/api/analyze` | POST | Predict mood from image |
| `/api/stylize` | POST | Apply style transfer |

### POST /api/analyze
```json
{ "image": "<base64 data URL>" }
```
Returns: `{ mood, confidence, scores, recommended_style, mood_description }`

### POST /api/stylize
```json
{
  "image": "<base64 data URL>",
  "style": "vangogh",
  "method": "neural",
  "strength": 1.0
}
```
Returns: `{ stylized_image, style, model, inference_time }`

---

## Model Details

**Mood Classifier**: VGGNet (VGG16 pretrained on ImageNet, fine-tuned on FER2013)
- Input: 224×224 RGB image
- Output: 5 mood classes (happy, calm, melancholic, energetic, dramatic)
- Training: Optuna hyperparameter search + cosine annealing LR scheduler

**Style Transfer**: Magenta arbitrary image stylization (Johnson et al. architecture)
- Single forward pass — ~1-3 seconds per image
- Supports style strength control (0.1 - 1.0)

---

## Git Workflow

This project follows git best practices:
- `main` — production-ready code only
- `develop` — integration branch
- `feature/*` — individual features
