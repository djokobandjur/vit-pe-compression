"""
Aggregate compression results into tables suitable for the paper.

Inputs: JSON outputs from baseline_eval, magnitude_pruning, structured_pruning,
ptq_quantization.

Outputs:
  - summary.csv with per-(PE, scope, ratio) means and stds across seeds
  - text-format tables printed to stdout
  - optional plots (accuracy vs ratio, by PE family)

Usage:
    python -m scripts.analyze_results \
        --baseline results/baseline/baseline_accuracy.json \
        --magnitude results/pruning/magnitude.json \
        --structured results/pruning/structured.json \
        --ptq results/quantization/ptq.json \
        --output_dir results/analysis
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


PE_TYPES = ["learned", "sinusoidal", "rope", "alibi"]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_by(data, group_keys, value_key="top1_accuracy"):
    """
    Group entries by group_keys (tuple of field names) and compute
    mean ± std of value_key across remaining variation (typically seeds).
    """
    bins = defaultdict(list)
    for k, v in data.items():
        if k.startswith("_"):
            continue
        try:
            group = tuple(v[g] for g in group_keys)
        except KeyError:
            continue
        bins[group].append(v[value_key])

    out = {}
    for g, vals in bins.items():
        arr = np.array(vals)
        out[g] = {"mean": arr.mean(), "std": arr.std(ddof=0), "n": len(arr)}
    return out


def print_baseline_table(baseline):
    print("\n" + "=" * 60)
    print("Baseline accuracy on ImageNet-100 val (mean ± std)")
    print("=" * 60)
    agg = aggregate_by(baseline, ("pe_type",))
    print(f"{'PE':<14}{'n':>4}{'mean':>10}{'std':>10}")
    for pe in PE_TYPES:
        if (pe,) in agg:
            a = agg[(pe,)]
            print(f"  {pe:<12}{a['n']:>4}{a['mean']:>10.4f}{a['std']:>10.4f}")


def print_pruning_table(pruning_data, title, ratio_key="ratio"):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    agg = aggregate_by(pruning_data, ("pe_type", "scope", ratio_key))
    scopes = sorted({k[1] for k in agg.keys()})
    ratios = sorted({k[2] for k in agg.keys()})
    for scope in scopes:
        print(f"\n[scope = {scope}]")
        header = f"{'PE':<14}" + "".join(f"{r:>9.2f}" for r in ratios)
        print(header)
        for pe in PE_TYPES:
            row = f"  {pe:<12}"
            for r in ratios:
                key = (pe, scope, r)
                if key in agg:
                    row += f"{agg[key]['mean']:>9.3f}"
                else:
                    row += f"{'--':>9}"
            print(row)


def print_structured_table(structured_data, title):
    """Structured pruning has two unit types (heads, neurons)."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    agg = aggregate_by(structured_data, ("pe_type", "unit", "scope", "ratio"))
    units = sorted({k[1] for k in agg.keys()})
    scopes = sorted({k[2] for k in agg.keys()})

    for unit in units:
        for scope in scopes:
            ratios = sorted({k[3] for k in agg.keys() if k[1] == unit and k[2] == scope})
            if not ratios:
                continue
            print(f"\n[unit = {unit}, scope = {scope}]")
            header = f"{'PE':<14}" + "".join(f"{r:>9.3f}" for r in ratios)
            print(header)
            for pe in PE_TYPES:
                row = f"  {pe:<12}"
                for r in ratios:
                    key = (pe, unit, scope, r)
                    if key in agg:
                        row += f"{agg[key]['mean']:>9.3f}"
                    else:
                        row += f"{'--':>9}"
                print(row)


def print_ptq_table(ptq_data, title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    agg = aggregate_by(ptq_data, ("pe_type", "scope", "bits"))
    scopes = sorted({k[1] for k in agg.keys()})
    bits = sorted({k[2] for k in agg.keys()}, reverse=True)  # 32, 8, 4, 2
    for scope in scopes:
        print(f"\n[scope = {scope}]")
        header = f"{'PE':<14}" + "".join(f"{f'INT{b}' if b < 32 else 'FP32':>9}" for b in bits)
        print(header)
        for pe in PE_TYPES:
            row = f"  {pe:<12}"
            for b in bits:
                key = (pe, scope, b)
                if key in agg:
                    row += f"{agg[key]['mean']:>9.3f}"
                else:
                    row += f"{'--':>9}"
            print(row)


def write_csv(data, output_path, group_keys, value_key="top1_accuracy"):
    """Write per-group mean/std CSV."""
    import csv
    agg = aggregate_by(data, group_keys, value_key=value_key)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(group_keys) + ["n", "mean", "std"])
        for k, v in sorted(agg.items()):
            w.writerow(list(k) + [v["n"], f"{v['mean']:.6f}", f"{v['std']:.6f}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--magnitude")
    parser.add_argument("--structured")
    parser.add_argument("--ptq")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.baseline:
        b = load_json(args.baseline)
        if b:
            print_baseline_table(b)
            write_csv(b, os.path.join(args.output_dir, "baseline.csv"),
                       ("pe_type",))

    if args.magnitude:
        d = load_json(args.magnitude)
        if d:
            print_pruning_table(d, "Magnitude pruning: top-1 accuracy (mean across seeds)")
            write_csv(d, os.path.join(args.output_dir, "magnitude.csv"),
                       ("pe_type", "scope", "ratio"))

    if args.structured:
        d = load_json(args.structured)
        if d:
            print_structured_table(d, "Structured pruning: top-1 accuracy (mean across seeds)")
            write_csv(d, os.path.join(args.output_dir, "structured.csv"),
                       ("pe_type", "unit", "scope", "ratio"))

    if args.ptq:
        d = load_json(args.ptq)
        if d:
            print_ptq_table(d, "Post-training quantization: top-1 accuracy (mean across seeds)")
            write_csv(d, os.path.join(args.output_dir, "ptq.csv"),
                       ("pe_type", "scope", "bits"))

    print(f"\n[DONE] CSVs written to {args.output_dir}")


if __name__ == "__main__":
    main()
