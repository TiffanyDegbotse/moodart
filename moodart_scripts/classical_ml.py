"""
classical_ml.py
---------------
Trains Random Forest and SVM classifiers using Optuna hyperparameter search.
Saves both pipelines and the best one to models/classifier/.

Usage:
    python scripts/classical_ml.py \
        --features data/processed \
        --output   models/classifier \
        --rf-trials 5 \
        --svm-trials 5
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Optuna objectives ─────────────────────────────────────────────────────────

def objective_rf(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    return accuracy_score(y_val, pipe.predict(X_val))


def objective_svm(trial, X_train, y_train, X_val, y_val):
    params = {
        'C':      trial.suggest_float('C', 0.1, 100.0, log=True),
        'gamma':  trial.suggest_categorical('gamma', ['scale', 'auto']),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
    }
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(**params, probability=True, class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return accuracy_score(y_val, pipe.predict(X_val))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train classical ML mood classifiers.')
    parser.add_argument('--features',      default='data/processed')
    parser.add_argument('--output',        default='models/classifier')
    parser.add_argument('--rf-trials',     type=int, default=5)
    parser.add_argument('--svm-trials',    type=int, default=5)
    parser.add_argument('--svm-subsample', type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs('data/outputs', exist_ok=True)

    # Load features
    X  = np.load(os.path.join(args.features, 'X.npy'))
    y  = np.load(os.path.join(args.features, 'y.npy'))
    with open(os.path.join(args.features, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

    print(f'Features: {X.shape} | Classes: {le.classes_}')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train)} | Val: {len(X_val)}')

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f'\nOptuna RF search ({args.rf_trials} trials)...')
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(
        lambda t: objective_rf(t, X_train, y_train, X_val, y_val),
        n_trials=args.rf_trials, show_progress_bar=True
    )
    print(f'Best RF val accuracy: {study_rf.best_value:.4f}')
    print(f'Best RF params: {study_rf.best_params}')

    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            **study_rf.best_params, class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf_pipeline.predict(X_val))
    print(f'Final RF Val Accuracy: {rf_val_acc:.4f}')
    print(classification_report(y_val, rf_pipeline.predict(X_val), target_names=le.classes_))

    # Optuna plots
    optuna.visualization.matplotlib.plot_optimization_history(study_rf)
    plt.title('Optuna RF Optimization History')
    plt.tight_layout()
    plt.savefig('data/outputs/optuna_rf_history.png', dpi=150)

    optuna.visualization.matplotlib.plot_param_importances(study_rf)
    plt.title('RF Hyperparameter Importances')
    plt.tight_layout()
    plt.savefig('data/outputs/optuna_rf_param_importance.png', dpi=150)

    # ── SVM ───────────────────────────────────────────────────────────────────
    X_sub, y_sub = resample(
        X_train, y_train,
        n_samples=min(args.svm_subsample, len(X_train)),
        random_state=42, stratify=y_train
    )
    print(f'\nOptuna SVM search ({args.svm_trials} trials, subsample={len(X_sub)})...')
    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(
        lambda t: objective_svm(t, X_sub, y_sub, X_val, y_val),
        n_trials=args.svm_trials, show_progress_bar=True
    )
    print(f'Best SVM val accuracy: {study_svm.best_value:.4f}')
    print(f'Best SVM params: {study_svm.best_params}')

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(**study_svm.best_params, probability=True,
                    class_weight='balanced', random_state=42))
    ])
    print('Training final SVM on full training data...')
    svm_pipeline.fit(X_train, y_train)
    svm_val_acc = accuracy_score(y_val, svm_pipeline.predict(X_val))
    print(f'Final SVM Val Accuracy: {svm_val_acc:.4f}')
    print(classification_report(y_val, svm_pipeline.predict(X_val), target_names=le.classes_))

    optuna.visualization.matplotlib.plot_optimization_history(study_svm)
    plt.title('Optuna SVM Optimization History')
    plt.tight_layout()
    plt.savefig('data/outputs/optuna_svm_history.png', dpi=150)

    optuna.visualization.matplotlib.plot_param_importances(study_svm)
    plt.title('SVM Hyperparameter Importances')
    plt.tight_layout()
    plt.savefig('data/outputs/optuna_svm_param_importance.png', dpi=150)

    # ── Select best and save ──────────────────────────────────────────────────
    if rf_val_acc >= svm_val_acc:
        best_pipeline, best_name = rf_pipeline, 'Random Forest'
    else:
        best_pipeline, best_name = svm_pipeline, 'SVM'

    print(f'\nBest classical model: {best_name}')

    with open(os.path.join(args.output, 'rf_pipeline.pkl'), 'wb') as f:
        pickle.dump(rf_pipeline, f)
    with open(os.path.join(args.output, 'svm_pipeline.pkl'), 'wb') as f:
        pickle.dump(svm_pipeline, f)
    with open(os.path.join(args.output, 'best_classical_pipeline.pkl'), 'wb') as f:
        pickle.dump(best_pipeline, f)

    print(f'Saved rf_pipeline.pkl, svm_pipeline.pkl, best_classical_pipeline.pkl to {args.output}')

    # Confusion matrix
    cm = confusion_matrix(y_val, best_pipeline.predict(X_val))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{best_name} Confusion Matrix (Validation)')
    plt.ylabel('True Mood')
    plt.xlabel('Predicted Mood')
    plt.tight_layout()
    plt.savefig('data/outputs/classical_confusion_matrix.png', dpi=150)
    print('Saved data/outputs/classical_confusion_matrix.png')


if __name__ == '__main__':
    main()