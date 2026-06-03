"""
Magnitude pruning evaluation.

Sets the smallest X% of weights (by absolute value) to zero and measures
the resulting top-1 accuracy on the ImageNet-100 val set. Pruning is
unstructured (individual weights), not structured (heads/neurons).

Four scopes are evaluated to disentangle which subspaces are essential:
  - global:        rank weights across the entire model, zero bottom X%
  - attention:     rank only inside attention layers (qkv, proj)
  - mlp:           rank only inside MLP layers (Sequential indices 0, 3)
  - per_layer:     within each transformer block, prune X% uniformly

PE-aware (secondary): a separate pass also targets the positional encoding
storage buffer that is consulted at inference time. The selection per PE
family:
  - learned:     pos_encoding.pos_embed (trainable parameter)
  - sinusoidal:  pos_encoding.pe        (registered buffer)
  - rope:        rope.cos_cached, rope.sin_cached (per-layer rotation tables)
  - alibi:       alibi.slopes           (per-head bias slopes)

The RoPE selection targets the cos/sin tables rather than the inv_freq
generator vector, because the forward pass consults the cos/sin tables
directly; the inv_freq vector is used only at module construction.

Pruning is non-destructive: each (model, ratio, scope) configuration starts
from a fresh copy of the original state_dict.

Output: JSON with accuracy curves per (pe_type, seed, ratio, scope).

Usage:
    python -m scripts.magnitude_pruning \\
        --checkpoint_root "/content/drive/MyDrive/Trained models_ImageNet100" \\
        --val_root /content/imagenet100/val \\
        --output_path .../results/pruning/magnitude.json
"""

import argparse
import copy
import json
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm

from models import load_pretrained_model, list_available_checkpoints
from data import get_imagenet100_val_loader


# ============================================================================
# Parameter selection by scope
# ============================================================================

def select_params(model, scope):
    """
    Return list of (name, param) pairs to be considered for pruning.

    Scopes:
      - global:           all weight matrices in attention + MLP
      - attention:        only qkv + proj inside MultiHeadAttention
      - mlp:              only the two Linear layers inside the MLP Sequential
      - per_layer:        same as global, but caller applies pruning per-block
      - pe_buffer_cache:  positional encoding storage buffer (see select_pe_tensors)
    """
    selected = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Skip biases, norms, classification head, cls_token, patch_embed
        if any(skip in name for skip in [".bias", "norm", "head", "cls_token", "patch_embed"]):
            continue
        # The PE buffer/parameter is handled separately under pe_buffer_cache
        if "pos_encoding" in name:
            continue

        is_attn = ".attn." in name and ("qkv" in name or "proj" in name)
        # MLP is nn.Sequential: index 0 (first Linear) and 3 (second Linear)
        is_mlp = ".mlp." in name and (".0." in name or ".3." in name)

        if scope == "global" or scope == "per_layer":
            if is_attn or is_mlp:
                selected.append((name, p))
        elif scope == "attention":
            if is_attn:
                selected.append((name, p))
        elif scope == "mlp":
            if is_mlp:
                selected.append((name, p))

    return selected


def select_pe_tensors(model):
    """
    Return the positional-encoding storage tensors that are consulted at
    inference time, per PE family.

    For RoPE we select cos_cached and sin_cached (the per-layer rotation
    tables used in the forward pass), not inv_freq (a deterministic
    construction vector that the forward pass does not consult).
    """
    selected = []
    pe_type = model.pe_type

    if pe_type == "learned":
        for name, p in model.named_parameters():
            if "pos_encoding.pos_embed" in name:
                selected.append((name, p))
    elif pe_type == "sinusoidal":
        for name, b in model.named_buffers():
            if "pos_encoding.pe" in name:
                selected.append((name, b))
    elif pe_type == "rope":
        for name, b in model.named_buffers():
            if "rope.cos_cached" in name or "rope.sin_cached" in name:
                selected.append((name, b))
    elif pe_type == "alibi":
        for name, b in model.named_buffers():
            if "alibi.slopes" in name:
                selected.append((name, b))

    return selected


# ============================================================================
# Pruning operations
# ============================================================================

def prune_global(tensors, ratio):
    """Zero out smallest |w| across ALL tensors combined."""
    if ratio <= 0:
        return
    all_abs = torch.cat([t.abs().flatten() for _, t in tensors])
    if all_abs.numel() == 0:
        return
    k = int(ratio * all_abs.numel())
    if k == 0:
        return
    threshold = all_abs.kthvalue(k).values.item()
    with torch.no_grad():
        for _, t in tensors:
            mask = t.abs() > threshold
            t.mul_(mask)


def prune_per_tensor(tensors, ratio):
    """Within each tensor, zero out smallest |w| independently."""
    if ratio <= 0:
        return
    with torch.no_grad():
        for _, t in tensors:
            flat = t.abs().flatten()
            k = int(ratio * flat.numel())
            if k == 0:
                continue
            threshold = flat.kthvalue(k).values.item()
            mask = t.abs() > threshold
            t.mul_(mask)


def prune_per_layer(model, ratio):
    """Apply per-tensor pruning to each transformer block independently."""
    if ratio <= 0:
        return
    for blk_idx, blk in enumerate(model.blocks):
        layer_tensors = []
        for name, p in blk.named_parameters():
            if not p.requires_grad:
                continue
            if any(skip in name for skip in [".bias", "norm"]):
                continue
            is_attn = ("attn" in name) and ("qkv" in name or "proj" in name)
            is_mlp = ("mlp" in name) and (".0." in name or ".3." in name)
            if is_attn or is_mlp:
                layer_tensors.append((f"block{blk_idx}.{name}", p))
        prune_per_tensor(layer_tensors, ratio)


# ============================================================================
# Evaluation loop
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def apply_pruning(model, scope, ratio):
    """Apply pruning in place. Returns dict of stats (params zeroed, etc.)."""
    if scope == "per_layer":
        prune_per_layer(model, ratio)
    elif scope == "pe_buffer_cache":
        tensors = select_pe_tensors(model)
        prune_per_tensor(tensors, ratio)
    else:
        tensors = select_params(model, scope)
        prune_global(tensors, ratio)

    # Count zeroed weights for verification
    zeroed = 0
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            zeroed += (p == 0).sum().item()
            total += p.numel()
    return {"zeroed_params": zeroed, "total_params": total,
            "actual_sparsity": zeroed / total if total else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--val_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pe_types", nargs="+",
                         default=["learned", "sinusoidal", "rope", "alibi"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--ratios", nargs="+", type=float,
                         default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--scopes", nargs="+",
                         default=["global", "attention", "mlp", "per_layer"],
                         help="Pruning scopes. Use 'pe_buffer_cache' for the "
                              "PE-aware secondary experiment that targets the "
                              "PE storage buffer consulted at inference time.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PE types: {args.pe_types}")
    print(f"[INFO] Seeds:    {args.seeds}")
    print(f"[INFO] Ratios:   {args.ratios}")
    print(f"[INFO] Scopes:   {args.scopes}")

    available = list_available_checkpoints(
        args.checkpoint_root, pe_types=args.pe_types, seeds=args.seeds,
    )
    print(f"[INFO] Checkpoints: {len(available)}")

    val_loader, val_dataset = get_imagenet100_val_loader(
        args.val_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"[INFO] Val set: {len(val_dataset)} images\n")

    results = OrderedDict()
    results["_metadata"] = {
        "ratios": args.ratios,
        "scopes": args.scopes,
        "val_size": len(val_dataset),
    }

    total_configs = len(available) * len(args.scopes) * len(args.ratios)
    pbar = tqdm(total=total_configs, desc="configs")

    for pe_type, seed in available:
        # Load checkpoint once, snapshot its state_dict for restoration.
        model = load_pretrained_model(
            args.checkpoint_root, pe_type, seed, device=device,
        )
        original_state = copy.deepcopy(model.state_dict())

        # For RoPE under pe_buffer_cache scope, the cos/sin caches are not in
        # the state_dict (they are stripped in model_loader.py because they
        # are regenerated deterministically at module construction). We
        # therefore snapshot them separately so we can restore them between
        # pruning configurations.
        cached_snapshot = None
        if pe_type == "rope" and "pe_buffer_cache" in args.scopes:
            cached_snapshot = {}
            for name, b in model.named_buffers():
                if "rope.cos_cached" in name or "rope.sin_cached" in name:
                    cached_snapshot[name] = b.clone()

        for scope in args.scopes:
            for ratio in args.ratios:
                # Restore original weights before each prune-evaluate cycle
                model.load_state_dict(original_state)
                if cached_snapshot is not None and scope == "pe_buffer_cache":
                    with torch.no_grad():
                        for name, b in model.named_buffers():
                            if name in cached_snapshot:
                                b.copy_(cached_snapshot[name])

                t0 = time.time()
                stats = apply_pruning(model, scope, ratio)
                acc = evaluate(model, val_loader, device)
                elapsed = time.time() - t0

                key = f"{pe_type}_seed{seed}__{scope}__r{int(ratio*100):02d}"
                results[key] = {
                    "pe_type": pe_type,
                    "seed": seed,
                    "scope": scope,
                    "ratio": ratio,
                    "top1_accuracy": acc,
                    "actual_sparsity": stats["actual_sparsity"],
                    "eval_time_s": elapsed,
                }
                pbar.set_postfix_str(
                    f"{pe_type[:4]}/s{seed}/{scope[:4]}/r={ratio:.1f} acc={acc:.3f}"
                )
                pbar.update(1)

                # Incremental save
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)

        del model, original_state
        torch.cuda.empty_cache()

    pbar.close()
    print(f"\n[DONE] Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
