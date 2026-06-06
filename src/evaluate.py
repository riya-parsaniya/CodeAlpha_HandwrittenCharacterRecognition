"""
evaluate.py
-----------
Evaluation: metrics, confusion matrix, training curves,
per-class accuracy, misclassification analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, top_k_accuracy_score
)

from config import PLOTS_DIR, CLASS_NAMES, NUM_CLASSES


def _ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ── 1. Metrics ────────────────────────────────────────────────────────────────

def print_metrics(model, X_test, y_test_oh, model_name='Model'):
    """y_test_oh: one-hot encoded labels."""
    y_true  = np.argmax(y_test_oh, axis=1)
    y_proba = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average='weighted')
    top3   = top_k_accuracy_score(y_true, y_proba, k=3)

    print(f'\n{"="*55}')
    print(f'  {model_name}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  Top-3 Acc: {top3:.4f}')
    print(f'  W-F1     : {f1:.4f}')
    print(f'{"="*55}')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    return {'accuracy': round(acc, 4), 'top3_accuracy': round(top3, 4),
            'f1_weighted': round(f1, 4)}


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test_oh,
                          model_name='Model', save=True):
    _ensure_dir()
    y_true = np.argmax(y_test_oh, axis=1)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm     = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        [f'{model_name} — Counts', f'{model_name} — Normalized']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.4, square=True,
                    cbar_kws={'shrink': 0.75}, annot_kws={'size': 7})
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual',    fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR,
                            f'confusion_{model_name.lower().replace(" ","_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[plot] Saved → {path}')
    plt.show()


# ── 3. Training Curves ────────────────────────────────────────────────────────

def plot_training_curves(history, model_name='Model', save=True):
    _ensure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, metric, title in zip(
        axes,
        [('accuracy', 'val_accuracy'), ('loss', 'val_loss')],
        ['Accuracy', 'Loss']
    ):
        ax.plot(history.history[metric[0]],     label='Train',
                color='#2563EB', lw=2)
        ax.plot(history.history[metric[1]], label='Validation',
                color='#DC2626', lw=2, linestyle='--')
        ax.set_title(f'{model_name} — {title}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR,
                            f'curves_{model_name.lower().replace(" ","_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[plot] Saved → {path}')
    plt.show()


# ── 4. Per-class Accuracy ─────────────────────────────────────────────────────

def plot_per_class_accuracy(model, X_test, y_test_oh,
                             model_name='Model', save=True):
    _ensure_dir()
    y_true  = np.argmax(y_test_oh, axis=1)
    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm      = confusion_matrix(y_true, y_pred)
    per_cls = cm.diagonal() / cm.sum(axis=1)

    colors = ['#16A34A' if v >= 0.85
              else '#F59E0B' if v >= 0.70
              else '#DC2626'
              for v in per_cls]

    plt.figure(figsize=(14, 5))
    bars = plt.bar(CLASS_NAMES, per_cls, color=colors,
                   edgecolor='white', width=0.65)
    for bar, val in zip(bars, per_cls):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.axhline(0.85, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.title(f'{model_name} — Per-class Accuracy (A–Z)',
              fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Character')
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR,
                            f'per_class_{model_name.lower().replace(" ","_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[plot] Saved → {path}')
    plt.show()


# ── 5. Misclassification Examples ────────────────────────────────────────────

def plot_misclassified(model, X_test, y_test_oh,
                       n=20, model_name='Model', save=True):
    _ensure_dir()
    y_true  = np.argmax(y_test_oh, axis=1)
    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    wrong   = np.where(y_true != y_pred)[0][:n]

    cols = 5
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()

    for ax, idx in zip(axes, wrong):
        ax.imshow(X_test[idx, :, :, 0], cmap='gray')
        ax.set_title(
            f'True: {CLASS_NAMES[y_true[idx]]}\n'
            f'Pred: {CLASS_NAMES[y_pred[idx]]}',
            fontsize=9, color='red'
        )
        ax.axis('off')
    for ax in axes[len(wrong):]:
        ax.axis('off')

    plt.suptitle(f'{model_name} — Misclassified Examples',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR,
                            f'misclassified_{model_name.lower().replace(" ","_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[plot] Saved → {path}')
    plt.show()


# ── 6. Model Comparison ───────────────────────────────────────────────────────

def plot_model_comparison(results: dict, save=True):
    """results = { 'ModelName': {'accuracy':..,'top3_accuracy':..,'f1_weighted':..} }"""
    _ensure_dir()
    import pandas as pd

    df = pd.DataFrame(results).T[['accuracy', 'top3_accuracy', 'f1_weighted']].astype(float)
    df.columns = ['Accuracy', 'Top-3 Accuracy', 'Weighted F1']

    ax = df.plot(kind='bar', figsize=(10, 5), colormap='Set2',
                 edgecolor='white', width=0.6)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.set_xticklabels(df.index, rotation=10, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, 'model_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'[plot] Saved → {path}')
    plt.show()
