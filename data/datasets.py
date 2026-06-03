"""
Dataset utilities for compression experiments.

Currently only ImageNet-100 val loader is needed. The val set must be
organized in ImageFolder layout: <root>/<wnid>/<image.JPEG>. Use
scripts/setup_imagenet100_val.py to produce this layout from the original
ILSVRC2012_img_val.tar archive.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet standard mean/std
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_val_transform(img_size=224):
    """Standard val-time transform: resize + center crop + normalize."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_imagenet100_val_loader(val_root, batch_size=128, num_workers=4, img_size=224):
    """
    Returns a DataLoader for the ImageNet-100 validation set.

    Args:
        val_root: path to ImageFolder root (e.g. /content/imagenet100/val)
        batch_size: minibatch size for evaluation
        num_workers: data loading workers
        img_size: input resolution (default 224 for ViT-Base/16)
    """
    transform = get_val_transform(img_size)
    dataset = datasets.ImageFolder(val_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset
