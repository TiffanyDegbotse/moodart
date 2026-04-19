.neural-gallery {
  margin-bottom: 1rem;
}

.neural-gallery-label {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: rgba(245, 240, 232, 0.4);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.75rem;
}

.neural-gallery-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.4rem;
  margin-bottom: 1rem;
}

.neural-gallery-item {
  cursor: pointer;
  border: 1px solid rgba(245, 240, 232, 0.1);
  border-radius: 2px;
  overflow: hidden;
  transition: all 0.2s;
  display: flex;
  flex-direction: column;
}

.neural-gallery-item:hover { border-color: var(--gold); }

.neural-gallery-item.selected {
  border-color: var(--gold);
  box-shadow: 0 0 0 1px var(--gold);
}

.neural-gallery-item img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  display: block;
}

.neural-gallery-item span {
  font-family: var(--font-mono);
  font-size: 0.55rem;
  color: rgba(245, 240, 232, 0.5);
  text-align: center;
  padding: 0.2rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.neural-note {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--gold);
  background: rgba(201, 168, 76, 0.08);
  border: 1px solid rgba(201, 168, 76, 0.2);
  border-radius: 2px;
  padding: 0.5rem 0.75rem;
  margin-bottom: 1rem;
  letter-spacing: 0.04em;
}
