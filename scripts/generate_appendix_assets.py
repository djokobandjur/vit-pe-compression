"""
generate_appendix_assets.py
============================
Generiše sve tabele za Appendix A i Appendix B iz JSON fajlova.
Pokretanje:
    python generate_appendix_assets.py --json_dir . --out_dir appendix_assets
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

PE_ORDER = ["rope", "sinusoidal", "alibi", "learned"]
PE_PRETTY = {
    "rope": "RoPE", "sinusoidal": "Sinusoidal",
    "alibi": "ALiBi", "learned": "Learned",
}


def load_all(json_dir: Path):
    return {
        "magnitude":    json.load(open(json_dir / "magnitude.json")),
        "magnitude_pe": json.load(open(json_dir / "magnitude_pe_buffer.json")),
        "structured":   json.load(open(json_dir / "structured.json")),
        "ptq":          json.load(open(json_dir / "ptq.json")),
        "ptq_pe":       json.load(open(json_dir / "ptq_pe_buffer.json")),
        "cka":          json.load(open(json_dir / "cka_pruning.json")),
    }


def aggregate(data, filter_fn):
    bucket = defaultdict(list)
    for k, v in data.items():
        if k == "_metadata": continue
        if filter_fn(v):
            bucket[v["pe_type"]].append(v["top1_accuracy"])
    return {pe: (float(np.mean(vs)), float(np.std(vs)))
            for pe, vs in bucket.items()}


def cell(mean, std):
    return f"{mean*100:.1f} $\\pm$ {std*100:.1f}"


# ============================================================
# Appendix A.1 — Magnitude Pruning (one table per scope)
# ============================================================
def a1_magnitude(data, out_dir: Path):
    mag = data["magnitude"]
    ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    scope_pretty = {"global": "Global", "attention": "Attention-only",
                    "mlp": "MLP-only", "per_layer": "Per-layer"}

    for scope in ["global", "attention", "mlp", "per_layer"]:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Magnitude pruning, " + scope_pretty[scope] + r" scope. "
            r"Top-$1$ accuracy (\%) on ImageNet-100, mean $\pm$ standard "
            r"deviation over three seeds.}",
            r"\label{tab:app_mag_" + scope + "}",
            r"\footnotesize",
            r"\begin{tabular}{l" + "c" * len(ratios) + "}",
            r"\toprule",
            r"PE family & " + " & ".join([f"$r{{=}}{r:.1f}$" for r in ratios])
            + r" \\",
            r"\midrule",
        ]
        for pe in PE_ORDER:
            cells = []
            for r in ratios:
                agg = aggregate(mag, lambda v: v["scope"] == scope
                                and abs(v["ratio"] - r) < 0.005)
                m, s = agg.get(pe, (0, 0))
                cells.append(cell(m, s))
            lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        out_path = out_dir / f"tab_app_mag_{scope}.tex"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"  → {out_path.name}")


# ============================================================
# Appendix A.2 — Structured Pruning (4 tables)
# ============================================================
def a2_structured(data, out_dir: Path):
    s = data["structured"]
    # heads: global + per_layer
    head_ratios = [0.0, 0.083, 0.167, 0.333, 0.5, 0.667]
    head_labels = ["0", "1/12", "2/12", "4/12", "6/12", "8/12"]
    # neurons: global + per_layer
    neuron_ratios = [0.0, 0.1, 0.25, 0.5, 0.75]

    for unit, ratios, labels in [
        ("heads", head_ratios, head_labels),
        ("neurons", neuron_ratios, [f"{r:.2f}" for r in neuron_ratios])
    ]:
        for scope in ["global", "per_layer"]:
            scope_pretty = {"global": "Global", "per_layer": "Per-layer"}[scope]
            unit_pretty = {"heads": "attention head", "neurons": "MLP neuron"}[unit]

            lines = [
                r"\begin{table}[H]",
                r"\centering",
                r"\caption{Structured " + unit_pretty + " pruning, "
                + scope_pretty + r" scope. Top-$1$ accuracy (\%) on "
                r"ImageNet-100, mean $\pm$ standard deviation over three "
                r"seeds.}",
                r"\label{tab:app_struct_" + unit + "_" + scope + "}",
                r"\footnotesize",
                r"\begin{tabular}{l" + "c" * len(ratios) + "}",
                r"\toprule",
                r"PE family & " + " & ".join(
                    [f"$r{{=}}{lab}$" for lab in labels]) + r" \\",
                r"\midrule",
            ]
            for pe in PE_ORDER:
                cells = []
                for r in ratios:
                    agg = aggregate(s, lambda v:
                                    v.get("unit") == unit
                                    and v.get("scope") == scope
                                    and abs(v.get("ratio", -1) - r) < 0.005)
                    m, sd = agg.get(pe, (0, 0))
                    cells.append(cell(m, sd))
                lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

            out_path = out_dir / f"tab_app_struct_{unit}_{scope}.tex"
            out_path.write_text("\n".join(lines) + "\n")
            print(f"  → {out_path.name}")


# ============================================================
# Appendix A.3 — Post-Training Quantization (3 tables)
# ============================================================
def a3_ptq(data, out_dir: Path):
    ptq = data["ptq"]
    bits = [32, 8, 4, 2]
    scope_pretty = {"global": "Global", "attention": "Attention-only",
                    "mlp": "MLP-only"}

    for scope in ["global", "attention", "mlp"]:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post-training quantization, " + scope_pretty[scope]
            + r" scope. Top-$1$ accuracy (\%) on ImageNet-100, mean $\pm$ "
            r"standard deviation over three seeds.}",
            r"\label{tab:app_ptq_" + scope + "}",
            r"\begin{tabular}{l" + "c" * len(bits) + "}",
            r"\toprule",
            r"PE family & " + " & ".join(
                [f"FP${b}$" if b == 32 else f"INT${b}$" for b in bits])
            + r" \\",
            r"\midrule",
        ]
        for pe in PE_ORDER:
            cells = []
            for b in bits:
                agg = aggregate(ptq, lambda v: v.get("scope") == scope
                                and v.get("bits") == b)
                m, sd = agg.get(pe, (0, 0))
                cells.append(cell(m, sd))
            lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        out_path = out_dir / f"tab_app_ptq_{scope}.tex"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"  → {out_path.name}")


# ============================================================
# Appendix A.4 — Positional Buffer compression
# ============================================================
def a4_buffer(data, out_dir: Path):
    # Magnitude pruning of PE buffer (full grid, all 7 ratios)
    mp = data["magnitude_pe"]
    ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Positional buffer magnitude pruning (full grid). Top-$1$ "
        r"accuracy (\%) on ImageNet-100, mean $\pm$ standard deviation over "
        r"three seeds.}",
        r"\label{tab:app_buf_mag}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(ratios) + "}",
        r"\toprule",
        r"PE family & " + " & ".join([f"$r{{=}}{r:.1f}$" for r in ratios])
        + r" \\",
        r"\midrule",
    ]
    for pe in PE_ORDER:
        cells = []
        for r in ratios:
            agg = aggregate(mp, lambda v:
                            abs(v.get("ratio", -1) - r) < 0.005)
            m, sd = agg.get(pe, (0, 0))
            cells.append(cell(m, sd))
        lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "tab_app_buf_mag.tex").write_text("\n".join(lines) + "\n")
    print(f"  → tab_app_buf_mag.tex")

    # PTQ of PE buffer (full grid, all 4 bits)
    pq = data["ptq_pe"]
    bits = [32, 8, 4, 2]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Positional buffer post-training quantization (full grid). "
        r"Top-$1$ accuracy (\%) on ImageNet-100, mean $\pm$ standard "
        r"deviation over three seeds.}",
        r"\label{tab:app_buf_ptq}",
        r"\begin{tabular}{l" + "c" * len(bits) + "}",
        r"\toprule",
        r"PE family & " + " & ".join(
            [f"FP${b}$" if b == 32 else f"INT${b}$" for b in bits])
        + r" \\",
        r"\midrule",
    ]
    for pe in PE_ORDER:
        cells = []
        for b in bits:
            agg = aggregate(pq, lambda v: v.get("bits") == b)
            m, sd = agg.get(pe, (0, 0))
            cells.append(cell(m, sd))
        lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "tab_app_buf_ptq.tex").write_text("\n".join(lines) + "\n")
    print(f"  → tab_app_buf_ptq.tex")


# ============================================================
# Appendix B — CKA grid
# ============================================================
def b_cka(data, out_dir: Path):
    cka = data["cka"]
    # Gather: scope, ratio combinations available
    combos = sorted({(v["scope"], v["ratio"])
                     for k, v in cka.items() if k != "_metadata"})

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Layer-wise linear CKA between original and compressed "
        r"models. Each cell is the mean CKA over three seeds; per-seed "
        r"standard deviations are below $0.02$ throughout. Computed on "
        r"$2{,}000$ ImageNet-100 validation images with stimulus seed $1$.}",
        r"\label{tab:app_cka_grid}",
        r"\footnotesize",
        r"\begin{tabular}{ll" + "c" * 12 + "}",
        r"\toprule",
        r"Scope, $r$ & PE family & " + " & ".join(
            [f"L{i}" for i in range(12)]) + r" \\",
        r"\midrule",
    ]
    last_combo = None
    for scope, ratio in combos:
        combo_label = f"{scope.replace('_', '-')}, $r{{=}}{ratio}$"
        first_row = True
        for pe in PE_ORDER:
            arrs = []
            for k, v in cka.items():
                if k == "_metadata": continue
                if (v["pe_type"] == pe and v["scope"] == scope
                        and abs(v["ratio"] - ratio) < 0.005):
                    arrs.append(v["cka_per_layer"])
            if not arrs:
                continue
            means = np.array(arrs).mean(axis=0)
            cells = [f"{m:.2f}" for m in means]
            label = combo_label if first_row else ""
            lines.append(f"{label} & {PE_PRETTY[pe]} & "
                         + " & ".join(cells) + r" \\")
            first_row = False
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "tab_app_cka_grid.tex").write_text("\n".join(lines) + "\n")
    print(f"  → tab_app_cka_grid.tex")


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", type=Path, default=Path("."))
    p.add_argument("--out_dir", type=Path, default=Path("appendix_assets"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(args.json_dir)

    print("Appendix A.1 — Magnitude Pruning:")
    a1_magnitude(data, args.out_dir)
    print("\nAppendix A.2 — Structured Pruning:")
    a2_structured(data, args.out_dir)
    print("\nAppendix A.3 — Post-Training Quantization:")
    a3_ptq(data, args.out_dir)
    print("\nAppendix A.4 — Positional Buffer:")
    a4_buffer(data, args.out_dir)
    print("\nAppendix B — CKA grid:")
    b_cka(data, args.out_dir)

    print(f"\nDone. Tables in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
