"""
generate_results_assets.py
==========================
Skripta koja iz JSON-ova generiše sve tabele i figure za Section 4.
Output: tablice u .tex fajlovima, figure u .pdf + .png.

Pokretanje:
    python generate_results_assets.py [--out_dir output_path]

Sve tabele i figure se generišu deterministički, sa konzistentnim stilom.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -------------------- Konstante --------------------

PE_ORDER = ["rope", "sinusoidal", "alibi", "learned"]
PE_PRETTY = {
    "rope": "RoPE",
    "sinusoidal": "Sinusoidal",
    "alibi": "ALiBi",
    "learned": "Learned",
}

# Paleta — konzistentna preko svih figura
PE_COLORS = {
    "rope":       "#1f77b4",  # plava
    "sinusoidal": "#ff7f0e",  # narandžasta
    "alibi":      "#2ca02c",  # zelena
    "learned":    "#d62728",  # crvena
}

PE_MARKERS = {
    "rope": "o",
    "sinusoidal": "s",
    "alibi": "^",
    "learned": "D",
}

# Publication style za matplotlib
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})


# -------------------- Učitavanje --------------------

def load_all(json_dir: Path):
    return {
        "baseline":      json.load(open(json_dir / "baseline_accuracy.json")),
        "magnitude":     json.load(open(json_dir / "magnitude.json")),
        "magnitude_pe":  json.load(open(json_dir / "magnitude_pe_buffer.json")),
        "structured":    json.load(open(json_dir / "structured.json")),
        "ptq":           json.load(open(json_dir / "ptq.json")),
        "ptq_pe":        json.load(open(json_dir / "ptq_pe_buffer.json")),
        "cka":           json.load(open(json_dir / "cka_pruning.json")),
    }


def aggregate(data, filter_fn):
    """Vrati {pe: (mean, std, n)} preko sva 3 seeda za stavke koje prolaze filter."""
    bucket = defaultdict(list)
    for k, v in data.items():
        if k == "_metadata":
            continue
        if filter_fn(v):
            bucket[v["pe_type"]].append(v["top1_accuracy"])
    return {pe: (float(np.mean(vs)), float(np.std(vs)), len(vs))
            for pe, vs in bucket.items()}


def cell(mean, std, bold=False, scale=100.0):
    """Formatiranje ćelije za LaTeX tabelu."""
    s = f"{mean*scale:.1f} $\\pm$ {std*scale:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


# -------------------- Tabele --------------------

def table_baseline(data, out: Path):
    bl = data["baseline"]
    rows = {}
    for pe in PE_ORDER:
        accs = [v["top1_accuracy"] for k, v in bl.items()
                if k != "_metadata" and v["pe_type"] == pe]
        rows[pe] = (np.mean(accs), np.std(accs))

    best = max(rows, key=lambda k: rows[k][0])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline top-$1$ accuracy (\%) on ImageNet-100 validation "
        r"before any compression, mean $\pm$ standard deviation over three "
        r"seeds. \textbf{Bold} marks the best PE family.}",
        r"\label{tab:baseline}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"PE family & Top-$1$ accuracy (\%) \\",
        r"\midrule",
    ]
    for pe in PE_ORDER:
        m, s = rows[pe]
        lines.append(f"{PE_PRETTY[pe]} & {cell(m, s, bold=(pe == best))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → {out.name}")


def table_magnitude_summary(data, out: Path):
    """Magnitude pruning u svim 4 scope-ovima, samo ratios 0.5 i 0.7."""
    mag = data["magnitude"]
    scopes = ["global", "attention", "mlp", "per_layer"]
    scope_pretty = {"global": "Global", "attention": "Attention",
                    "mlp": "MLP", "per_layer": "Per-layer"}
    ratios = [0.5, 0.7]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Magnitude pruning top-$1$ accuracy (\%) at moderate "
        r"($r{=}0.5$) and aggressive ($r{=}0.7$) sparsity ratios, mean $\pm$ "
        r"standard deviation over three seeds, for each of four pruning "
        r"scopes. The MLP scope at $r{=}0.7$ produces the cleanest PE-family "
        r"separation. Full grid in Appendix~\ref{app:full_grid}.}",
        r"\label{tab:magnitude_summary}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "cc" * len(scopes) + "}",
        r"\toprule",
        r" & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{{scope_pretty[s]}}}"
                              for s in scopes]) + r" \\",
        r" & " + " & ".join([f"$r{{=}}{r:.1f}$" for s in scopes for r in ratios]) + r" \\",
        r"\midrule",
    ]
    for pe in PE_ORDER:
        cells = []
        for s in scopes:
            for r in ratios:
                agg = aggregate(mag, lambda v: v["scope"] == s
                                and abs(v["ratio"] - r) < 0.01)
                m, sd, _ = agg.get(pe, (0, 0, 0))
                cells.append(cell(m, sd))
        lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → {out.name}")


def table_heads_global(data, out: Path):
    """Globalni head pruning, svi ratios."""
    s = data["structured"]
    ratios = [0.0, 0.083, 0.167, 0.333, 0.5, 0.667]
    ratio_labels = ["0", "1/12", "2/12", "4/12", "6/12", "8/12"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Global structured head pruning top-$1$ accuracy (\%), mean "
        r"$\pm$ standard deviation over three seeds. Head ratios are reported "
        r"as fractions of the $12$ attention heads per block. At $r{=}4/12$, "
        r"ALiBi retains $65.2\%$ while RoPE drops to $2.5\%$, a $26\times$ "
        r"ratio. \textbf{Bold} marks the best PE family per column.}",
        r"\label{tab:heads_global}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(ratios) + "}",
        r"\toprule",
        r"PE family & " + " & ".join([f"$r{{=}}{rl}$" for rl in ratio_labels]) + r" \\",
        r"\midrule",
    ]

    means = defaultdict(dict)
    stds = defaultdict(dict)
    for r in ratios:
        agg = aggregate(s, lambda v: v.get("unit") == "heads"
                        and v.get("scope") == "global"
                        and abs(v.get("ratio", -1) - r) < 0.005)
        for pe in PE_ORDER:
            m, sd, _ = agg.get(pe, (0, 0, 0))
            means[pe][r] = m
            stds[pe][r] = sd

    best_per_ratio = {r: max(PE_ORDER, key=lambda pe: means[pe][r]) for r in ratios}

    for pe in PE_ORDER:
        cells = [cell(means[pe][r], stds[pe][r], bold=(pe == best_per_ratio[r]))
                 for r in ratios]
        lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → {out.name}")


def table_ptq_buffer(data, out: Path):
    """PTQ na PE bufferima."""
    ptq_pe = data["ptq_pe"]
    bits = [32, 8, 4, 2]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-training quantization of positional buffers only, "
        r"top-$1$ accuracy (\%), mean $\pm$ standard deviation over three "
        r"seeds. RoPE and Sinusoidal buffers are essentially unaffected at "
        r"every bit width including INT$2$; ALiBi and Learned buffers "
        r"degrade more sharply, with ALiBi losing $7.6$pp at INT$2$. "
        r"\textbf{Bold} marks the best PE family per column.}",
        r"\label{tab:ptq_buffer}",
        r"\begin{tabular}{l" + "c" * len(bits) + "}",
        r"\toprule",
        r"PE family & " + " & ".join([f"INT${b}$" if b != 32 else f"FP${b}$"
                                       for b in bits]) + r" \\",
        r"\midrule",
    ]

    means = defaultdict(dict)
    stds = defaultdict(dict)
    for b in bits:
        agg = aggregate(ptq_pe, lambda v: v.get("bits") == b)
        for pe in PE_ORDER:
            m, sd, _ = agg.get(pe, (0, 0, 0))
            means[pe][b] = m
            stds[pe][b] = sd

    best_per_b = {b: max(PE_ORDER, key=lambda pe: means[pe][b]) for b in bits}

    for pe in PE_ORDER:
        cells = [cell(means[pe][b], stds[pe][b], bold=(pe == best_per_b[b]))
                 for b in bits]
        lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → {out.name}")


def table_pe_buffer_pruning(data, out: Path):
    """Magnitude pruning na PE bufferima."""
    mp = data["magnitude_pe"]
    ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Magnitude pruning of positional buffers only, top-$1$ "
        r"accuracy (\%), mean $\pm$ standard deviation over three seeds. "
        r"RoPE and Sinusoidal buffers degrade sharply above $r{=}0.3$, "
        r"with RoPE collapsing to $14.8\%$ and Sinusoidal to $5.5\%$ at "
        r"$r{=}0.9$. ALiBi and Learned buffers tolerate aggressive pruning, "
        r"losing at most $7.5$pp through $r{=}0.9$. \textbf{Bold} marks the "
        r"best PE family per column.}",
        r"\label{tab:pe_buffer_pruning}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(ratios) + "}",
        r"\toprule",
        r"PE family & " + " & ".join([f"$r{{=}}{r:.1f}$" for r in ratios]) + r" \\",
        r"\midrule",
    ]

    means = defaultdict(dict)
    stds = defaultdict(dict)
    for r in ratios:
        agg = aggregate(mp, lambda v: abs(v.get("ratio", -1) - r) < 0.005)
        for pe in PE_ORDER:
            m, sd, _ = agg.get(pe, (0, 0, 0))
            means[pe][r] = m
            stds[pe][r] = sd

    best_per_r = {r: max(PE_ORDER, key=lambda pe: means[pe][r]) for r in ratios}

    for pe in PE_ORDER:
        cells = [cell(means[pe][r], stds[pe][r], bold=(pe == best_per_r[r]))
                 for r in ratios]
        lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → {out.name}")


def table_cka_summary(data, out: Path):
    """CKA per-layer pri MLP r=0.7, srednje vrednosti preko seedova."""
    cka = data["cka"]
    rows_pe = defaultdict(list)
    for k, v in cka.items():
        if k == "_metadata":
            continue
        if v["scope"] == "mlp" and abs(v["ratio"] - 0.7) < 0.01:
            rows_pe[v["pe_type"]].append(v["cka_per_layer"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Layer-wise CKA between original and MLP-pruned models at "
        r"$r{=}0.7$, mean over three seeds (per-seed standard deviation "
        r"omitted; all are below $0.02$). All PE families show monotonic "
        r"decay to a minimum at layer~$10$, with a partial recovery at "
        r"layer~$11$.}",
        r"\label{tab:cka_mlp07}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * 12 + "}",
        r"\toprule",
        r"PE family & " + " & ".join([f"L{i}" for i in range(12)]) + r" \\",
        r"\midrule",
    ]
    for pe in PE_ORDER:
        arrs = np.array(rows_pe[pe])
        means = arrs.mean(axis=0)
        cells = [f"{m:.2f}" for m in means]
        lines.append(f"{PE_PRETTY[pe]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → {out.name}")


# -------------------- Figure --------------------

def fig_heads_pruning(data, out_pdf: Path, out_png: Path):
    """Two-panel: global vs per_layer head pruning."""
    s = data["structured"]
    ratios = [0.0, 0.083, 0.167, 0.333, 0.5, 0.667]
    ratio_labels = ["0", "1/12", "2/12", "4/12", "6/12", "8/12"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    for ax, scope, title in [(axes[0], "global", "Global scope"),
                              (axes[1], "per_layer", "Per-layer scope")]:
        for pe in PE_ORDER:
            means, stds = [], []
            for r in ratios:
                agg = aggregate(s, lambda v: v.get("unit") == "heads"
                                and v.get("scope") == scope
                                and abs(v.get("ratio", -1) - r) < 0.005)
                m, sd, _ = agg.get(pe, (0, 0, 0))
                means.append(m * 100)
                stds.append(sd * 100)
            xs = np.arange(len(ratios))
            ax.errorbar(xs, means, yerr=stds, label=PE_PRETTY[pe],
                        color=PE_COLORS[pe], marker=PE_MARKERS[pe],
                        capsize=3, capthick=1)
        ax.set_xticks(np.arange(len(ratios)))
        ax.set_xticklabels(ratio_labels)
        ax.set_xlabel("Head pruning ratio")
        ax.set_title(title)
        ax.set_ylim(-2, 95)
    axes[0].set_ylabel("Top-1 accuracy (\\%)")
    axes[1].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_pdf.name} + {out_png.name}")


def fig_pe_buffer(data, out_pdf: Path, out_png: Path):
    """Two-panel: PE buffer pod magnitude (lefti panel) i PTQ (desni panel)."""
    mp = data["magnitude_pe"]
    pq = data["ptq_pe"]
    ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    bits = [32, 8, 4, 2]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel 1: magnitude
    for pe in PE_ORDER:
        means, stds = [], []
        for r in ratios:
            agg = aggregate(mp, lambda v: abs(v.get("ratio", -1) - r) < 0.005)
            m, sd, _ = agg.get(pe, (0, 0, 0))
            means.append(m * 100)
            stds.append(sd * 100)
        axes[0].errorbar(ratios, means, yerr=stds, label=PE_PRETTY[pe],
                         color=PE_COLORS[pe], marker=PE_MARKERS[pe],
                         capsize=3, capthick=1)
    axes[0].set_xlabel("Magnitude pruning ratio")
    axes[0].set_ylabel("Top-1 accuracy (\\%)")
    axes[0].set_title("PE buffer magnitude pruning")
    axes[0].set_ylim(-2, 90)

    # Panel 2: PTQ (logaritamska osa za bits)
    for pe in PE_ORDER:
        means, stds = [], []
        for b in bits:
            agg = aggregate(pq, lambda v: v.get("bits") == b)
            m, sd, _ = agg.get(pe, (0, 0, 0))
            means.append(m * 100)
            stds.append(sd * 100)
        axes[1].errorbar(bits, means, yerr=stds, label=PE_PRETTY[pe],
                         color=PE_COLORS[pe], marker=PE_MARKERS[pe],
                         capsize=3, capthick=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(bits)
    axes[1].set_xticklabels([f"INT{b}" if b != 32 else "FP32" for b in bits])
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Quantization precision")
    axes[1].set_title("PE buffer quantization")
    axes[1].set_ylim(70, 87)
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_pdf.name} + {out_png.name}")


def fig_cka_depth(data, out_pdf: Path, out_png: Path):
    """CKA per-layer, jedan panel sa 4 PE × 3 ratios."""
    cka = data["cka"]
    ratios_show = [0.3, 0.5, 0.7]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)

    for ax, r in zip(axes, ratios_show):
        for pe in PE_ORDER:
            arrs = []
            for k, v in cka.items():
                if k == "_metadata":
                    continue
                if (v["pe_type"] == pe and v["scope"] == "mlp"
                        and abs(v["ratio"] - r) < 0.01):
                    arrs.append(v["cka_per_layer"])
            arrs = np.array(arrs)
            means = arrs.mean(axis=0)
            stds = arrs.std(axis=0)
            xs = np.arange(12)
            ax.errorbar(xs, means, yerr=stds, label=PE_PRETTY[pe],
                        color=PE_COLORS[pe], marker=PE_MARKERS[pe],
                        capsize=2, capthick=1, markersize=4)
        ax.set_xlabel("Transformer block")
        ax.set_title(f"MLP pruning $r{{=}}{r}$")
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Linear CKA")
    axes[-1].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_pdf.name} + {out_png.name}")


# -------------------- Main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", type=Path, default=Path("."))
    p.add_argument("--out_dir", type=Path, default=Path("results_assets"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(args.json_dir)

    print("Tables:")
    table_baseline(data,            args.out_dir / "tab_baseline.tex")
    table_magnitude_summary(data,   args.out_dir / "tab_magnitude_summary.tex")
    table_heads_global(data,        args.out_dir / "tab_heads_global.tex")
    table_ptq_buffer(data,          args.out_dir / "tab_ptq_buffer.tex")
    table_pe_buffer_pruning(data,   args.out_dir / "tab_pe_buffer_pruning.tex")
    table_cka_summary(data,         args.out_dir / "tab_cka_mlp07.tex")

    print("Figures:")
    fig_heads_pruning(data,  args.out_dir / "fig_heads_pruning.pdf",
                              args.out_dir / "fig_heads_pruning.png")
    fig_pe_buffer(data,      args.out_dir / "fig_pe_buffer.pdf",
                              args.out_dir / "fig_pe_buffer.png")
    fig_cka_depth(data,      args.out_dir / "fig_cka_depth.pdf",
                              args.out_dir / "fig_cka_depth.png")

    print(f"\nDone. All assets in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
