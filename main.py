"""Entry point: train or predict chess board states."""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import ChessRedDataset
from data.transforms import get_train_transforms, get_val_transforms
from inference.predict import load_model, predict
from model.loss import ChessLoss
from model.model import build_model
from training.callbacks import EarlyStopping, MetricsLogger, ModelCheckpoint
from training.plot import plot_training_curves
from training.train import train_epoch
from training.validate import validate


def _device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def run_train(cfg: dict) -> None:
    device = _device()
    img_size = cfg['data']['img_size']

    train_ds = ChessRedDataset(
        cfg['data']['dataroot'], split='train',
        transform=get_train_transforms(img_size))
    val_ds = ChessRedDataset(
        cfg['data']['dataroot'], split='val',
        transform=get_val_transforms(img_size))

    train_loader = DataLoader(
        train_ds, batch_size=cfg['training']['batch_size'], shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'])
    val_loader = DataLoader(
        val_ds, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'])

    model = build_model(cfg).to(device)

    criterion = ChessLoss(
        w_corners=cfg['loss']['w_corners'],
        w_occ=cfg['loss']['w_occupancy'],
        w_orient=cfg['loss']['w_orient'])

    opt_cfg = cfg['training']['optimizer']
    optimizer = AdamW(model.parameters(),
                      lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'])

    sched_cfg = cfg['training']['scheduler']
    scheduler = CosineAnnealingLR(optimizer,
                                   T_max=sched_cfg['t_max'],
                                   eta_min=sched_cfg['eta_min'])

    ckpt_cfg = cfg['training']['checkpoint']
    checkpoint_cb = ModelCheckpoint(
        save_dir=ckpt_cfg['save_dir'],
        monitor=ckpt_cfg['monitor'],
        mode=ckpt_cfg['mode'])
    early_stop_cb = EarlyStopping(
        monitor=ckpt_cfg['monitor'],
        patience=cfg['training']['early_stopping']['patience'],
        mode=ckpt_cfg['mode'])
    metrics_logger = MetricsLogger(
        save_path=f"{ckpt_cfg['save_dir']}/metrics.csv")

    use_amp = cfg['training']['mixed_precision'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"Device: {device}  |  train={len(train_ds)}  val={len(val_ds)}")

    for epoch in range(1, cfg['training']['epochs'] + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl = validate(model, val_loader, criterion, device)
        scheduler.step()

        metrics = {**tr, **vl}
        checkpoint_cb(model, optimizer, epoch, metrics)
        metrics_logger(epoch, metrics)

        print(
            f"Epoch {epoch:3d}  "
            f"loss={tr['loss']:.4f}  "
            f"train_acc={tr['train_exact_acc']:.4f}  "
            f"val_acc={vl['val_exact_acc']:.4f}  "
            f"val_tol1={vl['val_tol1_acc']:.4f}"
        )

        if early_stop_cb(metrics):
            break

    print(f"\nBest {checkpoint_cb.monitor}: {checkpoint_cb.best_value:.4f}")
    print(f"Checkpoint saved at: {checkpoint_cb.best_path}")

    plot_training_curves(
        csv_path=f"{ckpt_cfg['save_dir']}/metrics.csv",
        save_dir=ckpt_cfg['save_dir'],
        experiment_name=ckpt_cfg['save_dir'].split('/')[-1],
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def run_predict(args: argparse.Namespace, cfg: dict) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = args.checkpoint or cfg['inference']['checkpoint']
    model = load_model(ckpt, device=device)
    result = predict(model, args.image, device=device,
                     img_size=cfg['data']['img_size'])
    print(f"FEN        : {result['fen']}")
    print(f"Orientation: {result['orientation']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Chess board state recognition')
    parser.add_argument('--config', default='configs/config.yaml',
                        help='Path to YAML config file')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('train', help='Train the model')

    pred_p = sub.add_parser('predict', help='Run inference on an image')
    pred_p.add_argument('--image', required=True, help='Path to input image')
    pred_p.add_argument('--checkpoint', default=None,
                        help='Override checkpoint path from config')

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.command == 'train':
        run_train(cfg)
    elif args.command == 'predict':
        run_predict(args, cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
