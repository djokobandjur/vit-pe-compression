"""
Setup ImageNet-100 validation set from ILSVRC2012_img_val.tar.

Filters the full ImageNet-1k val set down to 100 classes (5000 images, 50 per
class) and organizes them in ImageFolder layout for use by downstream
evaluation scripts.

Usage:
    python -m scripts.setup_imagenet100_val \
        --tar_path /path/to/ILSVRC2012_img_val.tar \
        --output_dir /content/imagenet100

Requires data/imagenet100_classes.txt and data/val_labels.txt to be present
in the repo (or override via --classes_path and --labels_path).
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = REPO_ROOT / "data" / "imagenet100_classes.txt"
DEFAULT_LABELS = REPO_ROOT / "data" / "val_labels.txt"

VAL_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/research/"
    "slim/datasets/imagenet_2012_validation_synset_labels.txt"
)


def download_val_labels(output_path):
    print(f"[INFO] Downloading val_labels.txt to {output_path}")
    urllib.request.urlretrieve(VAL_LABELS_URL, output_path)


def load_text_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_path", required=True,
                         help="Path to ILSVRC2012_img_val.tar")
    parser.add_argument("--output_dir", required=True,
                         help="Output directory (will create <output_dir>/val/<wnid>/)")
    parser.add_argument("--classes_path", default=str(DEFAULT_CLASSES),
                         help="Path to imagenet100_classes.txt (default: repo data/)")
    parser.add_argument("--labels_path", default=str(DEFAULT_LABELS),
                         help="Path to val_labels.txt (default: repo data/, auto-downloads if missing)")
    args = parser.parse_args()

    print("=" * 60)
    print("ImageNet-100 Setup")
    print("=" * 60)

    # Check inputs
    if not os.path.exists(args.tar_path):
        raise FileNotFoundError(f"Tar archive not found: {args.tar_path}")
    print(f"  Found: {os.path.basename(args.tar_path)}")

    if not os.path.exists(args.classes_path):
        raise FileNotFoundError(f"Classes file not found: {args.classes_path}")
    print(f"  Found: {os.path.basename(args.classes_path)}")

    if not os.path.exists(args.labels_path):
        print(f"  Missing: {os.path.basename(args.labels_path)} -- downloading")
        os.makedirs(os.path.dirname(args.labels_path), exist_ok=True)
        download_val_labels(args.labels_path)
    print(f"  Found: {os.path.basename(args.labels_path)}")

    classes = set(load_text_lines(args.classes_path))
    labels = load_text_lines(args.labels_path)
    print(f"\nImageNet-100 classes: {len(classes)}")
    print(f"Val labels loaded:    {len(labels)} entries")

    # Create output structure
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    for wnid in classes:
        os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)
    print(f"Output directory:     {val_dir}")
    print(f"Created {len(classes)} class folders")

    # Build mapping: filename -> wnid
    # ImageNet val filenames are ILSVRC2012_val_00000001.JPEG ... 00050000
    fname_to_wnid = {}
    for i, wnid in enumerate(labels):
        fname = f"ILSVRC2012_val_{i+1:08d}.JPEG"
        if wnid in classes:
            fname_to_wnid[fname] = wnid

    # Extract only the relevant images
    print(f"\nExtracting from: {args.tar_path}")
    print("(This may take 5-10 minutes...)")

    extracted = 0
    skipped = 0
    with tarfile.open(args.tar_path, "r") as tar:
        members = tar.getmembers()
        print(f"Total images in tar: {len(members):,}")
        for m in tqdm(members, desc="Filtering"):
            fname = os.path.basename(m.name)
            if fname in fname_to_wnid:
                wnid = fname_to_wnid[fname]
                dest = os.path.join(val_dir, wnid, fname)
                if not os.path.exists(dest):
                    f = tar.extractfile(m)
                    if f is not None:
                        with open(dest, "wb") as out:
                            out.write(f.read())
                extracted += 1
            else:
                skipped += 1

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print(f"  Images extracted: {extracted:,}")
    print(f"  Images skipped:   {skipped:,}")
    print(f"  Expected:         5,000")

    if extracted == 5000:
        print("  All 5,000 images extracted successfully.")
    else:
        print(f"  WARNING: Expected 5,000 but got {extracted}")

    # Verify per-class count
    print("\nPer-class image count (first 5):")
    counts = []
    for wnid in sorted(classes):
        cnt = len(os.listdir(os.path.join(val_dir, wnid)))
        counts.append(cnt)
    for wnid, cnt in zip(sorted(classes)[:5], counts[:5]):
        print(f"  {wnid}: {cnt} images")
    print(f"\n  Min images per class: {min(counts)}")
    print(f"  Max images per class: {max(counts)}")
    if min(counts) == max(counts) == 50:
        print("All classes have exactly 50 images.")

    print(f"\nDataset ready at: {val_dir}")


if __name__ == "__main__":
    main()
