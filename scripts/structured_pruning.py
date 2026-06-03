"""
Structured pruning evaluation.

Removes whole units instead of individual weights:
  - heads:   zero out a fraction of attention heads (per layer or globally),
             ranked by L2 norm of the head's output projection columns
  - neurons: zero out a fraction of MLP hidden neurons, ranked by L2 norm of
             corresponding rows in the first Linear (mlp.0.weight)

Both produce real compute speedups when followed by reshape/contraction
(not done here -- we only zero them out and measure accuracy degradation).

Output: JSON with accuracy curves per (pe_type, seed, unit_type, ratio,
selection_scope).

Usage:
    python -m scripts.structured_pruning \
        --checkpoint_root "/content/drive/MyDrive/Trained models_ImageNet100" \
        --val_root /content/imagenet100/val \
        --output_path .../results/pruning/structured.json
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
# Importance scores
# ============================================================================

def head_importance(model):
    """
    Compute L2 norm of each attention head's contribution.

    For each transformer block, the output projection `attn.proj.weight` is
    (embed_dim, embed_dim) and conceptually splits into num_heads column
    groups of size head_dim. The norm of a column group reflects how much
    the corresponding head contributes to the block's output.

    Returns: dict {block_idx: tensor of shape (num_heads,)} with importance scores.
    """
    importances = {}
    for i, blk in enumerate(model.blocks):
        proj_w = blk.attn.proj.weight.data       # (embed_dim, embed_dim)
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim
        # Reshape to (embed_dim, num_heads, head_dim) so each head occupies
        # `head_dim` consecutive input columns
        proj_reshaped = proj_w.view(proj_w.size(0), num_heads, head_dim)
        # L2 norm per head over (output_dim, head_dim)
        scores = proj_reshaped.norm(dim=(0, 2))   # (num_heads,)
        importances[i] = scores.cpu()
    return importances


def neuron_importance(model):
    """
    Compute L2 norm of each MLP hidden neuron's input weights.

    `mlp.0.weight` is (hidden_dim, embed_dim); each row corresponds to one
    hidden neuron. Norm of a row reflects neuron's input sensitivity.

    Returns: dict {block_idx: tensor of shape (hidden_dim,)}.
    """
    importances = {}
    for i, blk in enumerate(model.blocks):
        # blk.mlp is nn.Sequential; index 0 is the first Linear
        fc1_w = blk.mlp[0].weight.data            # (hidden_dim, embed_dim)
        scores = fc1_w.norm(dim=1)                # (hidden_dim,)
        importances[i] = scores.cpu()
    return importances


# ============================================================================
# Pruning operations
# ============================================================================

def prune_heads(model, ratio, scope="per_layer"):
    """
    Zero out a fraction of attention heads.

    scope='per_layer': in each block, remove `ratio * num_heads` lowest-
                       importance heads (uniform per layer).
    scope='global':    rank all heads across all blocks; remove the lowest
                       `ratio * (depth * num_heads)` globally.
    """
    if ratio <= 0:
        return {"heads_pruned": 0, "total_heads": 0}

    importances = head_importance(model)
    total_heads = sum(t.numel() for t in importances.values())

    if scope == "per_layer":
        with torch.no_grad():
            pruned = 0
            for i, blk in enumerate(model.blocks):
                scores = importances[i]
                num_heads = blk.attn.num_heads
                head_dim = blk.attn.head_dim
                k = int(ratio * num_heads)
                if k == 0:
                    continue
                # Lowest-k head indices
                _, idx = scores.topk(k, largest=False)
                for h in idx.tolist():
                    _zero_head(blk, h, head_dim)
                    pruned += 1
    elif scope == "global":
        # Flatten (block_idx, head_idx, score), sort by score
        triples = []
        for i, scores in importances.items():
            for h, s in enumerate(scores.tolist()):
                triples.append((i, h, s))
        triples.sort(key=lambda x: x[2])
        k = int(ratio * len(triples))
        with torch.no_grad():
            for i, h, _ in triples[:k]:
                blk = model.blocks[i]
                _zero_head(blk, h, blk.attn.head_dim)
            pruned = k
    else:
        raise ValueError(f"Unknown scope: {scope}")

    return {"heads_pruned": pruned, "total_heads": total_heads}


def _zero_head(blk, head_idx, head_dim):
    """Zero rows/cols corresponding to one head in qkv and proj."""
    h = head_idx
    d = head_dim
    embed_dim = blk.attn.embed_dim

    # qkv.weight is (3*embed_dim, embed_dim): Q rows [0:embed_dim], K rows
    # [embed_dim:2*embed_dim], V rows [2*embed_dim:3*embed_dim]. Within each
    # third, head h occupies rows [h*d:(h+1)*d].
    qkv_w = blk.attn.qkv.weight.data
    qkv_b = blk.attn.qkv.bias.data
    for offset in (0, embed_dim, 2 * embed_dim):
        qkv_w[offset + h * d: offset + (h + 1) * d, :] = 0.0
        qkv_b[offset + h * d: offset + (h + 1) * d] = 0.0

    # proj.weight is (embed_dim, embed_dim); head h occupies input columns [h*d:(h+1)*d]
    blk.attn.proj.weight.data[:, h * d: (h + 1) * d] = 0.0


def prune_neurons(model, ratio, scope="per_layer"):
    """
    Zero out a fraction of MLP hidden neurons.

    For each pruned neuron, we zero the corresponding row in mlp.0.weight
    (and bias) and the corresponding column in mlp.3.weight.
    """
    if ratio <= 0:
        return {"neurons_pruned": 0, "total_neurons": 0}

    importances = neuron_importance(model)
    total_neurons = sum(t.numel() for t in importances.values())

    if scope == "per_layer":
        pruned = 0
        with torch.no_grad():
            for i, blk in enumerate(model.blocks):
                scores = importances[i]
                hidden_dim = scores.numel()
                k = int(ratio * hidden_dim)
                if k == 0:
                    continue
                _, idx = scores.topk(k, largest=False)
                fc1_w = blk.mlp[0].weight.data
                fc1_b = blk.mlp[0].bias.data
                fc2_w = blk.mlp[3].weight.data
                for n in idx.tolist():
                    fc1_w[n, :] = 0.0
                    fc1_b[n] = 0.0
                    fc2_w[:, n] = 0.0
                pruned += k
    elif scope == "global":
        triples = []
        for i, scores in importances.items():
            for n, s in enumerate(scores.tolist()):
                triples.append((i, n, s))
        triples.sort(key=lambda x: x[2])
        k = int(ratio * len(triples))
        with torch.no_grad():
            for i, n, _ in triples[:k]:
                blk = model.blocks[i]
                blk.mlp[0].weight.data[n, :] = 0.0
                blk.mlp[0].bias.data[n] = 0.0
                blk.mlp[3].weight.data[:, n] = 0.0
            pruned = k
    else:
        raise ValueError(f"Unknown scope: {scope}")

    return {"neurons_pruned": pruned, "total_neurons": total_neurons}


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
    parser.add_argument("--head_ratios", nargs="+", type=float,
                         default=[0.0, 0.083, 0.167, 0.333, 0.5, 0.667],
                         help="Fractions of heads to prune (default 0, 1, 2, 4, 6, 8 of 12)")
    parser.add_argument("--neuron_ratios", nargs="+", type=float,
                         default=[0.0, 0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--scopes", nargs="+", default=["per_layer", "global"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PE types: {args.pe_types}")
    print(f"[INFO] Head ratios:   {args.head_ratios}")
    print(f"[INFO] Neuron ratios: {args.neuron_ratios}")
    print(f"[INFO] Scopes:        {args.scopes}")

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
        "head_ratios": args.head_ratios,
        "neuron_ratios": args.neuron_ratios,
        "scopes": args.scopes,
        "val_size": len(val_dataset),
    }

    n_configs = (len(available) * len(args.scopes)
                  * (len(args.head_ratios) + len(args.neuron_ratios)))
    pbar = tqdm(total=n_configs, desc="configs")

    for pe_type, seed in available:
        model = load_pretrained_model(
            args.checkpoint_root, pe_type, seed, device=device,
        )
        original_state = copy.deepcopy(model.state_dict())

        for scope in args.scopes:
            # Heads
            for ratio in args.head_ratios:
                model.load_state_dict(original_state)
                t0 = time.time()
                stats = prune_heads(model, ratio, scope=scope)
                acc = evaluate(model, val_loader, device)
                elapsed = time.time() - t0
                key = f"{pe_type}_seed{seed}__heads__{scope}__r{int(ratio*1000):03d}"
                results[key] = {
                    "pe_type": pe_type, "seed": seed,
                    "unit": "heads", "scope": scope, "ratio": ratio,
                    "top1_accuracy": acc,
                    "units_pruned": stats["heads_pruned"],
                    "total_units": stats["total_heads"],
                    "eval_time_s": elapsed,
                }
                pbar.set_postfix_str(
                    f"{pe_type[:4]}/s{seed}/heads/{scope[:4]}/r={ratio:.2f} acc={acc:.3f}"
                )
                pbar.update(1)
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)

            # Neurons
            for ratio in args.neuron_ratios:
                model.load_state_dict(original_state)
                t0 = time.time()
                stats = prune_neurons(model, ratio, scope=scope)
                acc = evaluate(model, val_loader, device)
                elapsed = time.time() - t0
                key = f"{pe_type}_seed{seed}__neurons__{scope}__r{int(ratio*100):02d}"
                results[key] = {
                    "pe_type": pe_type, "seed": seed,
                    "unit": "neurons", "scope": scope, "ratio": ratio,
                    "top1_accuracy": acc,
                    "units_pruned": stats["neurons_pruned"],
                    "total_units": stats["total_neurons"],
                    "eval_time_s": elapsed,
                }
                pbar.set_postfix_str(
                    f"{pe_type[:4]}/s{seed}/neurons/{scope[:4]}/r={ratio:.2f} acc={acc:.3f}"
                )
                pbar.update(1)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)

        del model, original_state
        torch.cuda.empty_cache()

    pbar.close()
    print(f"\n[DONE] Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
