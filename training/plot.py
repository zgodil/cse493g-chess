"""Generate training curve plots from a metrics CSV."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(csv_path: str, save_dir: str, experiment_name: str = '') -> None:
    """Read per-epoch metrics CSV and save a 3-panel training curve figure.

    Panels:
        1. Total loss  (train + val)
        2. Exact-match accuracy  (train + val)
        3. Val tolerance-1 accuracy
    """
    df = pd.read_csv(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = df['epoch']
    title_prefix = f'{experiment_name} — ' if experiment_name else ''

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{title_prefix}Training Curves', fontsize=13)

    # --- Loss ---
    axes[0].plot(epochs, df['loss'],     label='Train', linewidth=1.5)
    axes[0].plot(epochs, df['val_loss'], label='Val',   linewidth=1.5, linestyle='--')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Exact-match accuracy ---
    axes[1].plot(epochs, df['train_exact_acc'], label='Train', linewidth=1.5)
    axes[1].plot(epochs, df['val_exact_acc'],   label='Val',   linewidth=1.5, linestyle='--')
    axes[1].set_title('Exact-Match Accuracy\n(all 64 squares correct)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Val tolerance-1 accuracy ---
    axes[2].plot(epochs, df['val_tol1_acc'], color='tab:green', linewidth=1.5)
    axes[2].set_title('Val Tolerance-1 Accuracy\n(≥63/64 squares correct)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / 'training_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [plot] saved training curves -> {out_path}')
