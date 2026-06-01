"""Resize all ChessReD images to 224x224 and save to a new directory.

Run once locally before uploading to Kaggle:
    python preprocess_dataset.py --src dataset --dst dataset_224

The resized dataset is functionally identical for training since the
DataLoader resizes to 224x224 anyway. Reduces ~23GB to ~1-2GB.
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def preprocess(src: Path, dst: Path, size: int = 224) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    # Copy annotations.json unchanged — paths are the same
    shutil.copy(src / 'annotations.json', dst / 'annotations.json')
    print("Copied annotations.json")

    # Collect all image paths from annotations
    with open(src / 'annotations.json') as f:
        data = json.load(f)

    images = data['images']
    print(f"Resizing {len(images)} images to {size}x{size}...")

    for img_info in tqdm(images, unit='img'):
        rel_path = Path(img_info['path'])           # e.g. images/0/G000_IMG000.jpg
        src_path = src / rel_path
        dst_path = dst / rel_path

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(src_path) as img:
            img = img.convert('RGB')
            img = img.resize((size, size), Image.LANCZOS)
            img.save(dst_path, 'JPEG', quality=90)

    print(f"\nDone. Resized dataset saved to: {dst}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='dataset',
                        help='Path to original ChessReD dataset')
    parser.add_argument('--dst', default='dataset_224',
                        help='Output path for resized dataset')
    parser.add_argument('--size', type=int, default=224,
                        help='Target image size (default: 224)')
    args = parser.parse_args()

    preprocess(Path(args.src), Path(args.dst), args.size)
