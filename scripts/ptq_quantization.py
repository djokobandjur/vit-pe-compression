"""
Post-Training Quantization (PTQ) evaluation.

Simulates fake quantization on weights at varying bit widths (8, 4, 2) using
per-tensor symmetric scale. No retraining is performed; weights are quantized
in place, evaluated on ImageNet-100 val, and then restored.

Scopes:
  - global:           quantize all weight matrices (attention + MLP)
  - attention:        quantize only attention layers
  - mlp:              quantize only MLP layers
  - pe_buffer_cache:  quantize the positional encoding storage buffer that
                      is consulted at inference time (see select_pe_tensors)

PE-aware secondary experiment. For each PE family the selection targets the
storage tensor used at forward-pass time:
  - learned:     pos_encoding.pos_embed (trainable parameter)
  - sinusoidal:  pos_encoding.pe        (registered buffer)
  - rope:        rope.cos_cached, rope.sin_cached (rotation tables)
  - alibi:       alibi.slopes           (per-head bias slopes)

The RoPE selection targets the cos/sin tables rather than inv_freq because
the forward pass consults the cos/sin tables directly; inv_freq is used
only at module construction.

Output: JSON with accuracy per (pe_type, seed, bit_width, scope).

Usage:
    python -m scripts.ptq_quantization \\
        --checkpoint_root "/content/drive/MyDrive/Trained models_ImageNet100" \\
        --val_root /content/imagenet100/val \\
        --output_path .../results/quantization/ptq.json
"""

import argparse
import copy
import json
import os
import time
from collections import OrderedDict

import torch
from tqdm import tqdm

from models import load_pretrained_model, list_available_checkpoints
from data import get_imagenet100_val_loader


# ============================================================================
# Fake quantization
# ============================================================================

def fake_quantize_tensor(t, n_bits, symmetric=True):
    """
    Quantize tensor to n_bits and dequantize back to original dtype.

    Per-tensor symmetric quantization:
        scale = max(|t|) / (2^(n_bits-1) - 1)
        q = round(t / scale).clamp(-Qmax, Qmax)
        t_hat = q * scale
    """
    if n_bits >= 32:
        return  # no-op
    qmax = (1 << (n_bits - 1)) - 1   # 127 for INT8, 7 for INT4, 1 for INT2
    if qmax < 1:
        qmax = 1
    with torch.no_grad():
        max_abs = t.abs().max().item()
        if max_abs == 0.0:
            return
        scale = max_abs / qmax
        q = torch.round(t / scale).clamp(-qmax, qmax)
        t.copy_(q * scale)


def select_weight_tensors(model, scope):
    """Return list of (name, tensor) pairs to quantize, matching scope."""
    selected = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(skip in name for skip in [".bias", "norm", "head", "cls_token", "patch_embed"]):
            continue
        if "pos_encoding" in name:
            # Handled separately under pe_buffer_cache scope
            continue

        is_attn = ".attn." in name and ("qkv" in name or "proj" in name)
        is_mlp = ".mlp." in name and (".0." in name or ".3." in name)

        if scope == "global":
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


def apply_quantization(model, scope, n_bits):
    """Quantize weights in place according to scope."""
    if scope == "pe_buffer_cache":
        tensors = select_pe_tensors(model)
    else:
        tensors = select_weight_tensors(model, scope)
    for _, t in tensors:
        fake_quantize_tensor(t, n_bits)
    return {"tensors_quantized": len(tensors)}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--val_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pe_types", nargs="+",
                         default=["learned", "sinusoidal", "rope", "alibi"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--bits", nargs="+", type=int, default=[32, 8, 4, 2],
                         help="Bit widths to evaluate (32 = no quantization baseline)")
    parser.add_argument("--scopes", nargs="+",
                         default=["global", "attention", "mlp"],
                         help="Quantization scopes. Use 'pe_buffer_cache' for "
                              "the PE-aware secondary experiment that targets "
                              "the PE storage buffer consulted at inference time.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PE types: {args.pe_types}")
    print(f"[INFO] Bits:     {args.bits}")
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
        "bits": args.bits, "scopes": args.scopes,
        "val_size": len(val_dataset),
    }

    n_configs = len(available) * len(args.scopes) * len(args.bits)
    pbar = tqdm(total=n_configs, desc="configs")

    for pe_type, seed in available:
        model = load_pretrained_model(
            args.checkpoint_root, pe_type, seed, device=device,
        )
        original_state = copy.deepcopy(model.state_dict())

        # For RoPE under pe_buffer_cache scope, the cos/sin caches are not in
        # the state_dict (they are stripped in model_loader.py because they
        # are regenerated deterministically at module construction). We
        # therefore snapshot them separately so we can restore between
        # quantization configurations.
        cached_snapshot = None
        if pe_type == "rope" and "pe_buffer_cache" in args.scopes:
            cached_snapshot = {}
            for name, b in model.named_buffers():
                if "rope.cos_cached" in name or "rope.sin_cached" in name:
                    cached_snapshot[name] = b.clone()

        for scope in args.scopes:
            for n_bits in args.bits:
                model.load_state_dict(original_state)
                if cached_snapshot is not None and scope == "pe_buffer_cache":
                    with torch.no_grad():
                        for name, b in model.named_buffers():
                            if name in cached_snapshot:
                                b.copy_(cached_snapshot[name])

                t0 = time.time()
                stats = apply_quantization(model, scope, n_bits)
                acc = evaluate(model, val_loader, device)
                elapsed = time.time() - t0
                key = f"{pe_type}_seed{seed}__{scope}__b{n_bits:02d}"
                results[key] = {
                    "pe_type": pe_type, "seed": seed,
                    "scope": scope, "bits": n_bits,
                    "top1_accuracy": acc,
                    "tensors_quantized": stats["tensors_quantized"],
                    "eval_time_s": elapsed,
                }
                pbar.set_postfix_str(
                    f"{pe_type[:4]}/s{seed}/{scope[:4]}/b={n_bits} acc={acc:.3f}"
                )
                pbar.update(1)
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)

        del model, original_state
        torch.cuda.empty_cache()

    pbar.close()
    print(f"\n[DONE] Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
