"""
CKA analysis on pruned models.

For each of the 12 pretrained ViT-Base checkpoints, we extract per-layer CLS
token features from (a) the unpruned model and (b) several pruned variants,
then compute layer-wise linear CKA between them.

The question this addresses: *where in the network does magnitude pruning
damage representations?* If CKA collapses earliest at a specific layer,
that layer is the compression bottleneck.

Linear CKA between two centered feature matrices X (n, d_x) and Y (n, d_y)
is computed as:

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

(equivalent to Kornblith et al. 2019 formulation for linear kernel)

Output: JSON with per-(pe_type, seed, scope, ratio, layer) CKA values.

Usage:
    python -m scripts.cka_pruning \
        --checkpoint_root "/content/drive/MyDrive/Trained models_ImageNet100" \
        --val_root /content/imagenet100/val \
        --output_path .../results/cka/cka_pruning.json
"""

import argparse
import copy
import json
import os
import random
import time
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models import load_pretrained_model, list_available_checkpoints
from data import get_imagenet100_val_loader

# Reuse pruning functions from magnitude_pruning so behavior is identical
from scripts.magnitude_pruning import apply_pruning


# ============================================================================
# Per-layer feature extraction (CLS token after each transformer block)
# ============================================================================

@torch.no_grad()
def extract_layer_features(model, loader, device, n_layers=12):
    """
    Run loader through model and return list of feature tensors, one per
    transformer block. Captures CLS token (index 0) after each block but
    BEFORE the final norm + head.

    Returns: list of length n_layers, each tensor (N, embed_dim).
    """
    model.eval()
    n_layers = len(model.blocks)
    feats_per_layer = [[] for _ in range(n_layers)]

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        # Manually run forward to capture each block's output
        z = model.patch_embed(x)
        cls = model.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = model.pos_encoding(z)
        z = model.pos_drop(z)
        for i, blk in enumerate(model.blocks):
            z = blk(z)
            # CLS token at position 0
            feats_per_layer[i].append(z[:, 0, :].cpu())

    return [torch.cat(fs, dim=0) for fs in feats_per_layer]


# ============================================================================
# Linear CKA
# ============================================================================

def center(X):
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X, Y):
    """
    Linear CKA between two feature matrices (n, d). Robust to differing d.
    """
    X = center(X.float())
    Y = center(Y.float())
    # Frobenius norm of cross-covariance squared
    xtY = X.T @ Y
    xtX = X.T @ X
    ytY = Y.T @ Y
    num = (xtY ** 2).sum()
    denom = (xtX ** 2).sum().sqrt() * (ytY ** 2).sum().sqrt()
    if denom == 0:
        return 0.0
    return (num / denom).item()


# ============================================================================
# Stimulus subset (fixed across all comparisons)
# ============================================================================

def build_stimulus_loader(val_root, n_stimuli, stimulus_seed, batch_size, num_workers):
    """Random subset of val set, deterministic given seed."""
    base_loader, dataset = get_imagenet100_val_loader(
        val_root, batch_size=batch_size, num_workers=num_workers,
    )
    rng = random.Random(stimulus_seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = sorted(indices[:n_stimuli])  # sort for stable order
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--val_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pe_types", nargs="+",
                         default=["learned", "sinusoidal", "rope", "alibi"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--ratios", nargs="+", type=float,
                         default=[0.3, 0.5, 0.7, 0.8],
                         help="Pruning ratios at which to compare to FP32 original")
    parser.add_argument("--scopes", nargs="+", default=["global", "mlp"])
    parser.add_argument("--n_stimuli", type=int, default=2000)
    parser.add_argument("--stimulus_seed", type=int, default=1,
                         help="RNG seed for stimulus selection (Springer used 0)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device:    {device}")
    print(f"[INFO] PE types:  {args.pe_types}")
    print(f"[INFO] Seeds:     {args.seeds}")
    print(f"[INFO] Ratios:    {args.ratios}")
    print(f"[INFO] Scopes:    {args.scopes}")
    print(f"[INFO] Stimuli:   {args.n_stimuli} (seed={args.stimulus_seed})")

    available = list_available_checkpoints(
        args.checkpoint_root, pe_types=args.pe_types, seeds=args.seeds,
    )
    print(f"[INFO] Checkpoints: {len(available)}")

    stimulus_loader = build_stimulus_loader(
        args.val_root, args.n_stimuli, args.stimulus_seed,
        args.batch_size, args.num_workers,
    )
    print(f"[INFO] Stimulus loader: {len(stimulus_loader.dataset)} images\n")

    results = OrderedDict()
    results["_metadata"] = {
        "ratios": args.ratios,
        "scopes": args.scopes,
        "n_stimuli": args.n_stimuli,
        "stimulus_seed": args.stimulus_seed,
        "metric": "linear_cka",
    }

    n_configs = len(available) * len(args.scopes) * len(args.ratios)
    pbar = tqdm(total=n_configs, desc="configs")

    for pe_type, seed in available:
        # Load model once; snapshot state for pruning rollback
        model = load_pretrained_model(
            args.checkpoint_root, pe_type, seed, device=device,
        )
        original_state = copy.deepcopy(model.state_dict())
        n_layers = len(model.blocks)

        # Step 1: extract original features (run once per model)
        t0 = time.time()
        orig_feats = extract_layer_features(model, stimulus_loader, device, n_layers)
        orig_time = time.time() - t0

        # Step 2: for each (scope, ratio), prune + extract + compute CKA
        for scope in args.scopes:
            for ratio in args.ratios:
                model.load_state_dict(original_state)
                apply_pruning(model, scope, ratio)

                t0 = time.time()
                pruned_feats = extract_layer_features(model, stimulus_loader, device, n_layers)
                extract_time = time.time() - t0

                # Per-layer CKA
                ckas = []
                for L in range(n_layers):
                    c = linear_cka(orig_feats[L], pruned_feats[L])
                    ckas.append(c)

                key = f"{pe_type}_seed{seed}__{scope}__r{int(ratio*100):02d}"
                results[key] = {
                    "pe_type": pe_type, "seed": seed,
                    "scope": scope, "ratio": ratio,
                    "cka_per_layer": ckas,
                    "min_cka": min(ckas),
                    "min_cka_layer": int(ckas.index(min(ckas))),
                    "extract_time_s": extract_time,
                }
                pbar.set_postfix_str(
                    f"{pe_type[:4]}/s{seed}/{scope[:4]}/r={ratio:.1f} "
                    f"min_cka={min(ckas):.3f}@L{ckas.index(min(ckas))}"
                )
                pbar.update(1)

                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)

        del model, original_state, orig_feats
        torch.cuda.empty_cache()

    pbar.close()
    print(f"\n[DONE] Results saved to {args.output_path}")

    # Quick summary
    print("\n" + "=" * 70)
    print("Most-damaged layer by (pe_type, scope, ratio):")
    print("=" * 70)
    print(f"{'config':<40}{'min CKA':>10}{'at layer':>10}")
    for k, v in results.items():
        if k.startswith("_"): continue
        print(f"  {k:<38}{v['min_cka']:>10.3f}{v['min_cka_layer']:>10}")


if __name__ == "__main__":
    main()
