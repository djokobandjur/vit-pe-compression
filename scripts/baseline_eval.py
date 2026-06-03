"""
Baseline evaluation: top-1 accuracy of each pretrained model on the
ImageNet-100 validation set. This produces the reference numbers against
which all compression experiments are compared.

Output: JSON file with {(pe_type, seed): top1_accuracy} entries plus
metadata (parameter count, total wall-clock time per evaluation).

Usage:
    python -m scripts.baseline_eval \
        --checkpoint_root "/content/drive/MyDrive/Trained models_ImageNet100" \
        --val_root /content/imagenet100/val \
        --output_path /content/drive/MyDrive/pe_compression_experiment/results/baseline/baseline_accuracy.json
"""

import argparse
import json
import os
import time
from collections import OrderedDict

import torch
from tqdm import tqdm

from models import load_pretrained_model, list_available_checkpoints
from data import get_imagenet100_val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute top-1 accuracy on the loader."""
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def count_parameters(model):
    """Count trainable + buffer parameters separately."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return trainable, buffers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True,
                         help="Root containing {pe_type}_seed{seed}/best_model.pth")
    parser.add_argument("--val_root", required=True,
                         help="ImageNet-100 val (ImageFolder layout)")
    parser.add_argument("--output_path", required=True,
                         help="JSON output path")
    parser.add_argument("--pe_types", nargs="+",
                         default=["learned", "sinusoidal", "rope", "alibi"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PE types: {args.pe_types}")
    print(f"[INFO] Seeds:    {args.seeds}")

    # Sanity-check checkpoint availability
    available = list_available_checkpoints(
        args.checkpoint_root, pe_types=args.pe_types, seeds=args.seeds
    )
    print(f"[INFO] Found {len(available)} checkpoints")
    if not available:
        raise SystemExit("No checkpoints found; check --checkpoint_root path")

    # Build val loader once (reused across models)
    print(f"[INFO] Loading val set from {args.val_root}")
    val_loader, val_dataset = get_imagenet100_val_loader(
        args.val_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Val set: {len(val_dataset)} images, {len(val_dataset.classes)} classes")

    results = OrderedDict()
    results["_metadata"] = {
        "val_root": args.val_root,
        "val_size": len(val_dataset),
        "num_classes_in_val": len(val_dataset.classes),
        "batch_size": args.batch_size,
        "device": device,
    }

    for pe_type, seed in available:
        key = f"{pe_type}_seed{seed}"
        print(f"\n[MODEL] {key}")
        t0 = time.time()
        model = load_pretrained_model(
            args.checkpoint_root, pe_type, seed,
            num_classes=args.num_classes, device=device,
        )
        load_time = time.time() - t0

        trainable, buffers = count_parameters(model)
        print(f"  Trainable params: {trainable/1e6:.2f}M, Buffer params: {buffers/1e6:.2f}M")

        t0 = time.time()
        acc = evaluate(model, val_loader, device)
        eval_time = time.time() - t0
        print(f"  Top-1 accuracy: {acc:.4f}  (eval: {eval_time:.1f}s, load: {load_time:.1f}s)")

        results[key] = {
            "pe_type": pe_type,
            "seed": seed,
            "top1_accuracy": acc,
            "trainable_params": trainable,
            "buffer_params": buffers,
            "eval_time_s": eval_time,
            "load_time_s": load_time,
        }

        # Free memory before next model
        del model
        torch.cuda.empty_cache()

        # Incremental save (resilient to disconnects)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Baseline accuracies (top-1 on ImageNet-100 val):")
    print("=" * 60)
    by_pe = {}
    for k, v in results.items():
        if k.startswith("_"):
            continue
        by_pe.setdefault(v["pe_type"], []).append(v["top1_accuracy"])
    for pe in ["learned", "sinusoidal", "rope", "alibi"]:
        if pe in by_pe:
            accs = by_pe[pe]
            mean = sum(accs) / len(accs)
            print(f"  {pe:12s} n={len(accs)}  mean={mean:.4f}  values={[f'{a:.4f}' for a in accs]}")

    print(f"\n[DONE] Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
