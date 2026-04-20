import React, { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './App.css';

const API = process.env.REACT_APP_API_URL || '';

// ── Constants ─────────────────────────────────────────────────────────────────
const STYLES = {
  monet:     { label: 'Monet',     subtitle: 'Water Lilies',    color: '#7eb8c9', mood: 'happy' },
  vangogh:   { label: 'Van Gogh',  subtitle: 'Starry Night',    color: '#2d4a8a', mood: 'melancholic' },
  kandinsky: { label: 'Kandinsky', subtitle: 'Composition 7',   color: '#c94b2d', mood: 'energetic' },
  hokusai:   { label: 'Hokusai',   subtitle: 'The Great Wave',  color: '#4a7c8a', mood: 'calm' },
  munch:     { label: 'Munch',     subtitle: 'The Scream',      color: '#8a6a2d', mood: 'dramatic' },
};

const MOOD_COLORS = {
  happy:       '#c9a84c',
  calm:        '#4a7c8a',
  melancholic: '#5a5a8a',
  energetic:   '#c94b2d',
  dramatic:    '#8a4a2d',
};

const MOOD_LABELS = {
  happy:       'Happy',
  calm:        'Calm',
  melancholic: 'Melancholic',
  energetic:   'Energetic',
  dramatic:    'Dramatic',
};

const DEFAULT_NEURAL_GALLERY = {
  monet:     '/neural_gallery/monet.png',
  vangogh:   '/neural_gallery/vangogh.png',
  kandinsky: '/neural_gallery/kandinsky.png',
  hokusai:   '/neural_gallery/hokusai.png',
  munch:     '/neural_gallery/munch.png',
};

// ── Sub-components ────────────────────────────────────────────────────────────
function GrainOverlay() {
  return (
    <svg className="grain" xmlns="http://www.w3.org/2000/svg">
      <filter id="grain">
        <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch"/>
        <feColorMatrix type="saturate" values="0"/>
      </filter>
      <rect width="100%" height="100%" filter="url(#grain)" opacity="0.04"/>
    </svg>
  );
}

function UploadZone({ onImageSelected, hasImage }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => onImageSelected(e.target.result);
    reader.readAsDataURL(file);
  }, [onImageSelected]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  return (
    <motion.div
      className={`upload-zone ${dragging ? 'dragging' : ''} ${hasImage ? 'has-image' : ''}`}
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
      <div className="upload-inner">
        <motion.div
          className="upload-icon"
          animate={{ y: dragging ? -8 : 0 }}
          transition={{ type: 'spring', stiffness: 300 }}
        >
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <rect x="8" y="8" width="32" height="32" rx="2" stroke="currentColor" strokeWidth="1.5"/>
            <circle cx="18" cy="20" r="3" stroke="currentColor" strokeWidth="1.5"/>
            <path d="M8 32l10-10 8 8 6-6 8 8" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
          </svg>
        </motion.div>
        <p className="upload-label">
          {dragging ? 'Release to upload' : 'Drop your image here'}
        </p>
        <p className="upload-sub">or click to browse</p>
      </div>
    </motion.div>
  );
}

function MoodBadge({ mood, confidence, description }) {
  const color = MOOD_COLORS[mood] || '#c9a84c';
  return (
    <motion.div
      className="mood-badge"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      style={{ '--mood-color': color }}
    >
      <div className="mood-badge-header">
        <span className="mood-label-text">Detected Mood</span>
        <span className="mood-confidence">{Math.round(confidence * 100)}% confidence</span>
      </div>
      <div className="mood-name">{MOOD_LABELS[mood] || mood}</div>
      <p className="mood-description">{description}</p>
    </motion.div>
  );
}

// ── Style Card with custom painting upload ────────────────────────────────────
function StyleCard({ styleKey, style, selected, recommended, onClick, customImage, onCustomImage }) {
  const inputRef = useRef();

  const handleCustomFile = useCallback((e) => {
    e.stopPropagation();
    const file = e.target.files[0];
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (ev) => onCustomImage(styleKey, ev.target.result);
    reader.readAsDataURL(file);
  }, [styleKey, onCustomImage]);

  const handleEditClick = useCallback((e) => {
    e.stopPropagation();
    inputRef.current.click();
  }, []);

  return (
    <motion.div
      className={`style-card ${selected ? 'selected' : ''} ${recommended ? 'recommended' : ''}`}
      style={{ '--style-color': style.color }}
      whileHover={{ y: -4 }}
    >
      {/* Painting preview thumbnail */}
      {customImage && (
        <div className="style-card-thumb" onClick={() => onClick(styleKey)}>
          <img src={customImage} alt={style.label} className="style-thumb-img" />
        </div>
      )}

      <div className="style-card-body" onClick={() => onClick(styleKey)}>
        {recommended && <span className="rec-badge">✦ Recommended</span>}
        <div className="style-dot" />
        <span className="style-name">{style.label}</span>
        <span className="style-sub">{style.subtitle}</span>
      </div>

      {/* Edit button — tap to swap painting for this mood */}
      <button
        className="style-edit-btn"
        onClick={handleEditClick}
        title={`Change painting for ${MOOD_LABELS[style.mood]}`}
      >
        ✎
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleCustomFile}
      />
    </motion.div>
  );
}

function MoodBar({ mood, score }) {
  return (
    <div className="mood-bar-row">
      <span className="mood-bar-label">{MOOD_LABELS[mood] || mood}</span>
      <div className="mood-bar-track">
        <motion.div
          className="mood-bar-fill"
          style={{ background: MOOD_COLORS[mood] }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.round(score * 100)}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      <span className="mood-bar-pct">{Math.round(score * 100)}%</span>
    </div>
  );
}

function ResultPanel({ original, result, style, method, customGallery }) {
  const [view, setView] = useState('split');
  const [selectedGalleryStyle, setSelectedGalleryStyle] = useState(style);

  const isNeural = method === 'neural';

  // For neural: use custom image if set, else default gallery
  const getStyleSrc = (styleName) =>
    customGallery[styleName] || DEFAULT_NEURAL_GALLERY[styleName];

  const stylizedSrc = isNeural
    ? getStyleSrc(selectedGalleryStyle)
    : `data:image/png;base64,${result.stylized_image}`;

  return (
    <motion.div
      className="result-panel"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="result-header">
        <div className="result-tabs">
          {['split', 'original', 'stylized'].map(v => (
            <button
              key={v}
              className={`result-tab ${view === v ? 'active' : ''}`}
              onClick={() => setView(v)}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
        <div className="result-meta">
          <span className="meta-tag">{result.model}</span>
          <span className="meta-tag">{result.inference_time}s</span>
        </div>
      </div>

      {isNeural && (
        <div className="neural-gallery">
          <p className="neural-gallery-label">Neural Style Transfer — Select a style:</p>
          <div className="neural-gallery-grid">
            {Object.entries(DEFAULT_NEURAL_GALLERY).map(([name]) => (
              <div
                key={name}
                className={`neural-gallery-item ${selectedGalleryStyle === name ? 'selected' : ''}`}
                onClick={() => setSelectedGalleryStyle(name)}
              >
                <img src={getStyleSrc(name)} alt={name} />
                <span>{STYLES[name]?.label || name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={`result-images view-${view}`}>
        {(view === 'split' || view === 'original') && (
          <div className="result-img-wrap">
            <img src={original} alt="Original" />
            <span className="img-label">Original</span>
          </div>
        )}
        {(view === 'split' || view === 'stylized') && (
          <div className="result-img-wrap">
            <img src={stylizedSrc} alt="Stylized" />
            <span className="img-label">
              {isNeural
                ? `Neural — ${STYLES[selectedGalleryStyle]?.label || selectedGalleryStyle}`
                : `${STYLES[style]?.label} Style`}
            </span>
          </div>
        )}
      </div>

      <a
        className="download-btn"
        href={stylizedSrc}
        download={`moodart_${isNeural ? selectedGalleryStyle : style}.png`}
      >
        Download Artwork ↓
      </a>
    </motion.div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [image, setImage] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedStyle, setSelectedStyle] = useState('vangogh');
  const [selectedMethod, setSelectedMethod] = useState('neural');
  const [stylizedResult, setStylizedResult] = useState(null);
  const [strength, setStrength] = useState(1.0);
  const [step, setStep] = useState('upload');
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [error, setError] = useState(null);

  // Custom paintings per style slot — starts empty, falls back to defaults
  const [customGallery, setCustomGallery] = useState({});

  const handleCustomImage = useCallback((styleKey, dataUrl) => {
    setCustomGallery(prev => ({ ...prev, [styleKey]: dataUrl }));
  }, []);

  const handleImageSelected = useCallback(async (dataUrl) => {
    setImage(dataUrl);
    setAnalysisResult(null);
    setStylizedResult(null);
    setError(null);
    setStep('analyze');
    setLoading(true);
    setLoadingMsg('Reading your image...');
    try {
      const res = await axios.post(`${API}/api/analyze`, { image: dataUrl });
      setAnalysisResult(res.data);
      setSelectedStyle(res.data.recommended_style);
      setStep('style');
    } catch (e) {
      setError('Analysis failed. Please try again.');
      setStep('upload');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleStylize = useCallback(async () => {
    if (!image || !selectedStyle) return;

    if (selectedMethod === 'neural') {
      setStylizedResult({
        model: 'Deep Learning (Fast NST — Magenta)',
        inference_time: 0,
        style: selectedStyle,
      });
      setStep('result');
      return;
    }

    // For kmeans/naive: if custom image set for this style, send it to backend
    const customStyleImage = customGallery[selectedStyle] || null;

    setLoading(true);
    setLoadingMsg('Painting your image...');
    setError(null);
    try {
      const res = await axios.post(`${API}/api/stylize`, {
        image,
        style: selectedStyle,
        method: selectedMethod,
        strength,
        custom_style_image: customStyleImage,
      });
      setStylizedResult(res.data);
      setStep('result');
    } catch (e) {
      setError('Style transfer failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [image, selectedStyle, selectedMethod, strength, customGallery]);

  const handleReset = () => {
    setImage(null);
    setAnalysisResult(null);
    setStylizedResult(null);
    setError(null);
    setStep('upload');
  };

  return (
    <div className="app">
      <GrainOverlay />

      <motion.header
        className="header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className="header-inner">
          <div className="logo">
            <span className="logo-mark">✦</span>
            <span className="logo-text">MoodArt</span>
          </div>
          <p className="tagline">AI reads your image — art transforms it</p>
        </div>
      </motion.header>

      <main className="main">
        <div className="steps">
          {['Upload', 'Analyze', 'Style', 'Result'].map((s, i) => {
            const stepKeys = ['upload', 'analyze', 'style', 'result'];
            const current = stepKeys.indexOf(step);
            return (
              <div key={s} className={`step-item ${i <= current ? 'done' : ''} ${i === current ? 'active' : ''}`}>
                <div className="step-dot">{i < current ? '✓' : i + 1}</div>
                <span className="step-label">{s}</span>
                {i < 3 && <div className={`step-line ${i < current ? 'done' : ''}`} />}
              </div>
            );
          })}
        </div>

        <AnimatePresence>
          {error && (
            <motion.div className="error-banner" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {error}
              <button onClick={() => setError(null)}>✕</button>
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {loading && (
            <motion.div className="loading-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="loading-spinner" />
              <p className="loading-msg">{loadingMsg}</p>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="content">
          <div className="left-col">
            <motion.section
              className="section"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <h2 className="section-title">
                <span className="section-num">01</span> Your Image
              </h2>
              {image ? (
                <div className="image-preview-wrap">
                  <img className="image-preview" src={image} alt="Uploaded" />
                  <button className="change-btn" onClick={handleReset}>✕ Change image</button>
                </div>
              ) : (
                <UploadZone onImageSelected={handleImageSelected} hasImage={!!image} />
              )}
            </motion.section>

            <AnimatePresence>
              {analysisResult && (
                <motion.section
                  className="section"
                  key="analysis"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <h2 className="section-title">
                    <span className="section-num">02</span> Mood Analysis
                    <span className="section-badge">{analysisResult.model}</span>
                  </h2>
                  <MoodBadge
                    mood={analysisResult.mood}
                    confidence={analysisResult.confidence}
                    description={analysisResult.mood_description}
                  />
                  <div className="mood-bars">
                    {Object.entries(analysisResult.scores)
                      .sort(([,a],[,b]) => b - a)
                      .map(([mood, score]) => (
                        <MoodBar key={mood} mood={mood} score={score} />
                      ))
                    }
                  </div>
                </motion.section>
              )}
            </AnimatePresence>
          </div>

          <div className="right-col">
            <AnimatePresence>
              {(step === 'style' || step === 'result') && (
                <motion.section
                  className="section"
                  key="style-section"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <h2 className="section-title">
                    <span className="section-num">03</span> Choose Style
                  </h2>

                  <p className="style-hint">Tap ✎ on any card to swap its painting</p>

                  <div className="style-grid">
                    {Object.entries(STYLES).map(([key, style]) => (
                      <StyleCard
                        key={key}
                        styleKey={key}
                        style={style}
                        selected={selectedStyle === key}
                        recommended={analysisResult?.recommended_style === key}
                        onClick={setSelectedStyle}
                        customImage={customGallery[key] || null}
                        onCustomImage={handleCustomImage}
                      />
                    ))}
                  </div>

                  <div className="method-row">
                    <span className="method-label">Method</span>
                    <div className="method-toggle">
                      {[
                        { key: 'neural',  label: 'Neural (Deep Learning)' },
                        { key: 'kmeans',  label: 'K-Means (Classical ML)' },
                        { key: 'naive',   label: 'Color LUT (Baseline)' },
                      ].map(m => (
                        <button
                          key={m.key}
                          className={`method-btn ${selectedMethod === m.key ? 'active' : ''}`}
                          onClick={() => setSelectedMethod(m.key)}
                        >
                          {m.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {selectedMethod === 'neural' && (
                    <motion.div className="neural-note" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                      ✦ Neural mode shows pre-rendered results — tap ✎ on a style card to swap its painting
                    </motion.div>
                  )}

                  <div className="strength-row">
                    <span className="method-label">Strength — {Math.round(strength * 100)}%</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={strength}
                      onChange={e => setStrength(parseFloat(e.target.value))}
                      className="strength-slider"
                    />
                  </div>

                  <motion.button
                    className="stylize-btn"
                    onClick={handleStylize}
                    disabled={loading}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {loading ? 'Painting...' : 'Apply Style Transfer →'}
                  </motion.button>
                </motion.section>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {stylizedResult && (
                <ResultPanel
                  key="result"
                  original={image}
                  result={stylizedResult}
                  style={stylizedResult.style}
                  method={selectedMethod}
                  customGallery={customGallery}
                />
              )}
            </AnimatePresence>

            {step === 'upload' && !loading && (
              <motion.div
                className="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <p className="empty-title">How it works</p>
                <ol className="how-list">
                  <li>Upload any photo</li>
                  <li>AI detects the emotional mood</li>
                  <li>Choose an artistic style (or use our recommendation)</li>
                  <li>Tap ✎ on any style to swap its painting</li>
                  <li>Download your artwork</li>
                </ol>
                <div className="model-info">
                  <p className="model-info-title">Three AI approaches</p>
                  <p>Naive baseline · K-Means palette · Neural style transfer</p>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>MoodArt · AIPI 540 Final Project · Built with VGGNet + Neural Style Transfer</p>
      </footer>
    </div>
  );
}
