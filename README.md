# Positional Encoding Determines Structural Redundancy in Vision Transformers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20527499.svg)](https://doi.org/10.5281/zenodo.20527499)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Code, reproducibility scripts, and full accuracy matrices for the paper
*"Positional Encoding Determines Structural Redundancy in Vision Transformers"*
(manuscript under submission).
---

## What is this repository?

Vision Transformers use a *positional encoding* (PE) to inject spatial
structure into a permutation-invariant attention mechanism. The companion
transfer paper showed that PE choice has measurable effects on in-domain
accuracy and downstream representations. This work asks a different
question: does PE choice also determine which parts of a trained ViT are
structurally essential — i.e. which components survive aggressive
compression and which collapse?

The twelve ViT-Base/16 checkpoints used in this study (4 PE families × 3
seeds, pretrained on ImageNet-100) are subjected to three
one-shot compression techniques applied without any fine-tuning:
unstructured magnitude pruning (4 scopes × 7 ratios), structured pruning
of attention heads and MLP neurons (2 scopes × 11 ratio settings),
post-training quantization (3 scopes × 4 bit widths), and two auxiliary
sweeps that target only the positional encoding buffer (magnitude pruning
and PTQ). Layer-wise linear CKA between original and compressed models
localizes where in the network compression damages representations. The
full grid covers 876 evaluation configurations.

The two main findings are:

| Claim | One-line summary |
| --- | --- |
| **C1 (primary)** | **ALiBi tolerates global head pruning that collapses RoPE**: removing one third of attention heads leaves ALiBi at 65.2% top-1 accuracy while RoPE drops to 2.5% — a 26× ratio at the same compression level and the largest single-treatment effect across the entire compression grid. |
| **C2 (primary)** | **Positional buffers partition into two storage-format classes with inverted robustness profiles.** The two storage formats we name *bounded periodic tables* (RoPE, Sinusoidal) and *heterogeneous parametric values* (ALiBi, Learned). The first tolerates INT2 quantization to within 1pp of baseline but collapses under 90% magnitude pruning (−70pp to −76pp); the second shows the opposite pattern, fragile under INT2 (up to −7.6pp) but robust to 90% pruning (within −7.5pp). |

A layer-wise CKA analysis of the MLP-pruned models shows that compression
damage accumulates with depth and reaches its minimum at block 10 of 12,
consistent with classifier-head proximity — distinct from the block-9
representational divergence peak reported by the companion transfer paper.
The two phenomena (parametric redundancy under compression, representational
divergence under different training choices) localize at different depths
and reflect different structural modalities of the same positional
inductive bias.

---

## Repository layout

This repository contains the **code** and the **complete accuracy matrices**
(seven JSON files summarizing all 876 compression evaluations plus
layer-wise CKA), together with the figures used in the paper. Large
binary artefacts — pretrained checkpoints (~3.8 GB) — are hosted on
Google Drive (see *Trained models* below).

```
vit-pe-compression/
├── README.md                       # this file
├── LICENSE                         # MIT (code) + CC BY 4.0 (results, figures)
├── .gitignore
│
├── data/
│   ├── __init__.py                 # exports get_imagenet100_val_loader
│   ├── datasets.py                 # DataLoader: ImageFolder + ImageNet transforms
│   ├── imagenet100_classes.txt     # 100 class IDs (same subset as transfer paper)
│   └── val_labels.txt              # 50k ILSVRC2012 val labels (used by setup script)
│ 
├── models/
│   ├── __init__.py                 # exports VisionTransformer + load_pretrained_model
│   ├── vit_architecture.py         # ViT-Base + 4 PE variants (Learned, Sinusoidal, RoPE, ALiBi)
│   └── model_loader.py             # loads .pth checkpoints into the right PE variant
│ 
├── notebooks/
│   ├── 01_compression_sanity_check.ipynb   # loads one checkpoint, runs a small pruning probe
│   └── 02_compression_main_workflow.ipynb  # end-to-end pipeline, Colab-friendly
│ 
├── scripts/
│   ├── setup_imagenet100_val.py    # build ImageNet-100 val/ from ILSVRC2012 tar
│   ├── baseline_eval.py            # uncompressed top-1 accuracy → baseline_accuracy.json
│   ├── magnitude_pruning.py        # unstructured pruning sweep → magnitude.json
│   ├── structured_pruning.py       # head + neuron pruning sweep → structured.json
│   ├── ptq_quantization.py         # post-training quantization sweep → ptq.json
│   ├── cka_pruning.py              # layer-wise CKA between original and pruned models
│   ├── analyze_results.py          # per-PE summary tables (CSVs)
│   ├── generate_results_assets.py  # main-text figures (PDF/PNG)
│   └── generate_appendix_assets.py # 14 appendix tables (LaTeX includes)
│
├── results/                        # complete accuracy matrices (876 configs)
│   ├── baseline/
│   │   └── baseline_accuracy.json
│   ├── pruning/
│   │   ├── magnitude.json                  # 12 × 4 scopes × 7 ratios = 336 configs
│   │   ├── magnitude_pe_buffer.json        # 12 × 7 ratios = 84 PE-buffer configs
│   │   └── structured.json                 # 12 × 2 scopes × (heads + neurons) = 264 configs
│   ├── quantization/
│   │   ├── ptq.json                        # 12 × 3 scopes × 4 bits = 144 configs
│   │   └── ptq_pe_buffer.json              # 12 × 4 bits = 48 PE-buffer configs
│   ├── cka/
│   │   └── cka_pruning.json                # layer-wise CKA grid (scope × ratio × PE × seed)
│   └── analysis/
│       └── *.csv                           # per-PE summary tables aggregated by analyze_results.py
│
└── figures/                        # 3 main-text figures (regenerated by generate_results_assets.py)
    ├── fig_heads_pruning.{pdf,png}
    ├── fig_pe_buffer.{pdf,png}
    └── fig_cka_depth.{pdf,png}
```

The `figures/` directory and any LaTeX `tab_*.tex` files used by the paper
are not committed to this repository; they are regenerated from the JSON
outputs by `generate_results_assets.py` and `generate_appendix_assets.py`.


**Not in this repository** (obtain separately):

- `<checkpoint_root>/` — 12 ViT-Base `.pth` checkpoints (~3.8 GB; Google
  Drive link in *Trained models* below).

### Script-to-output mapping

| Script | Purpose | Output |
| --- | --- | --- |
| `setup_imagenet100_val.py` | Extracts the 100 selected classes from the ILSVRC2012 validation tar into an ImageFolder-compatible `val/` directory (50 images per class). | `<output_dir>/val/<class>/` |
| `baseline_eval.py` | Top-1 accuracy of each pretrained checkpoint on the ImageNet-100 validation set, before any compression. | `results/baseline/baseline_accuracy.json` |
| `magnitude_pruning.py` | Unstructured magnitude pruning sweep. Four scopes (global, attention, MLP, per-layer); seven ratios; optionally `pe_buffer_cache` scope for the PE-buffer secondary experiment. | `results/pruning/magnitude.json`, `results/pruning/magnitude_pe_buffer.json` |
| `structured_pruning.py` | Structured pruning of attention heads (L2-norm importance) and MLP neurons (input-row L2 norm). Two scopes (global, per-layer); six head ratios and five neuron ratios. | `results/pruning/structured.json` |
| `ptq_quantization.py` | Post-training quantization with per-tensor symmetric scale. Three scopes (global, attention, MLP); four bit widths (32, 8, 4, 2); optionally `pe_buffer_cache` scope. | `results/quantization/ptq.json`, `results/quantization/ptq_pe_buffer.json` |
| `cka_pruning.py` | Layer-wise linear CKA between original and pruned models on a fixed 2000-image stimulus subset of ImageNet-100 val. Two scopes (global, MLP); four ratios. | `results/cka/cka_pruning.json` |
| `analyze_results.py` | Aggregates the JSON outputs into per-PE summary CSVs. | `results/analysis/*.csv` |
| `generate_results_assets.py` | Produces the three main-text figures used in the paper from the JSON outputs. | `figures/fig_*.{pdf,png}` |
| `generate_appendix_assets.py` | Produces the 14 LaTeX include files for paper Appendix A (magnitude pruning, structured pruning, PTQ) and Appendix B (layer-wise CKA grid). Expects all JSON outputs in a flat directory. | `<out_dir>/tab_app_*.tex` |

---

## Trained models

The twelve ViT-Base checkpoints used in this study were trained by the
authors (4 PE families × 3 seeds, all trained from scratch on
ImageNet-100 for 300 epochs at 224×224 with an identical recipe; the
only thing that differs between checkpoints is the positional-encoding
implementation and the random seed). The full set (~3.8 GB) is hosted
on Google Drive:

- **Our ImageNet-100 ViT-Base models:** 
  [Google Drive folder (~3.8 GB)](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9)
  — public access, no Google account required.

The folder layout expected by `model_loader.py` is:

```
<models_dir>/
├── learned_seed42/best_model.pth
├── learned_seed123/best_model.pth
├── learned_seed456/best_model.pth
├── sinusoidal_seed42/best_model.pth
├── ...
└── alibi_seed456/best_model.pth
```

The code, accuracy matrices, and figures in this repository are archived on Zenodo (see DOI badge at the top of this README) for a stable, citable record. The checkpoint files themselves remain on Google Drive due to size (~3.8 GB).. The checkpoint files themselves remain on Google Drive due
to size (~3.8 GB).

---

## Prerequisites

Before running the notebook or the CLI workflow, anyone reproducing these
results needs to obtain two things and set one path. Result folders
(`results/`, `figures/`, …) are created automatically on first run — no
manual `mkdir` needed.

**1. The twelve ViT-Base checkpoints (required)**

The checkpoints are hosted on Google Drive — see the *Trained models*
section above for the link. Place the unpacked folder anywhere with
enough disk space (~3.8 GB) and use that path as `<CHECKPOINT_ROOT>`
(CLI) or `CHECKPOINT_ROOT` (notebook). The expected internal layout is
documented in *Trained models*.

**2. ImageNet-1k validation tar (required)**

The class list (`data/imagenet100_classes.txt`) and the full ILSVRC2012
val label index (`data/val_labels.txt`) are shipped in this repository,
so the only path you need to set is where the tar lives. In the notebook
this is the `IMAGENET_TAR` variable; on the CLI it is `--tar_path` to
`setup_imagenet100_val.py`.

The class list (`data/imagenet100_classes.txt`) is shipped in this
repository, so the only path you need to set is where the tar lives.
In the notebook this is the `IMAGENET_TAR` variable; on the CLI it is
`--tar_path` to `setup_imagenet100_val.py`.

**3. One path decision: `<DATA_HOME>`**

Pick one persistent directory where the JSON outputs and regenerated
figures will be written. In Colab this is typically your Google Drive
(the default in the notebook is
`/content/drive/MyDrive/pe_compression_experiment`); for local execution
it can be any folder. The notebook reads this from the `RESULTS_ROOT`
variable in the 1.4 path-constants cell. The CLI uses explicit
`--output_path` flags per step.

---

## Reproducing the results

### Quick start (notebook)

The fastest path is
[`notebooks/02_compression_main_workflow.ipynb`](notebooks/02_compression_main_workflow.ipynb).
It clones this repo, mounts Drive for checkpoint and dataset access, and
walks through:

1. Baseline accuracy on ImageNet-100 val (12 models)
2. Magnitude pruning sweep (12 × 4 scopes × 7 ratios = 336 configs)
3. Structured pruning sweep (12 × 2 scopes × 11 ratio settings = 264 configs)
4. Post-training quantization sweep (12 × 3 scopes × 4 bits = 144 configs)
5. PE-buffer secondary experiments (magnitude + PTQ on positional buffer only, 132 configs)
6. Layer-wise CKA on MLP-pruned models
7. Aggregate analysis (per-PE summary CSVs)
8. Regeneration of paper-ready PDF figures

For a minimal smoke test that only loads one checkpoint and verifies the
environment, use
[`notebooks/01_compression_sanity_check.ipynb`](notebooks/01_compression_sanity_check.ipynb).

Both notebooks have a header section titled **🔧 Configuration required**
that lists the small number of paths (`CHECKPOINT_ROOT`, `RESULTS_ROOT`,
`IMAGENET_TAR`) you must edit before running. Cells that require editing
are flagged inline with `>>> USER CONFIGURATION REQUIRED <<<` banners; all
other cells use derived paths or ephemeral `/content/` storage and need
no changes.

### Manual CLI workflow

The notebook is a thin wrapper over CLI commands. All scripts are invoked
as Python modules from the repository root; run `python -m scripts.<name> --help`
for the per-script flags. The command sequence below reproduces all files
shipped under `results/`. Replace `<CHECKPOINT_ROOT>`, `<RESULTS_ROOT>`,
and `<IMAGENET_VAL_ROOT>` with paths on your machine.

**1. Prepare ImageNet-100 validation set**

```bash
python -m scripts.setup_imagenet100_val \
    --tar_path     "/path/to/ILSVRC2012_img_val.tar" \
    --classes_path "data/imagenet100_classes.txt" \
    --output_dir   "<IMAGENET_VAL_ROOT>"
```

**2. Baseline accuracy**

```bash
python -m scripts.baseline_eval \
    --checkpoint_root "<CHECKPOINT_ROOT>" \
    --val_root        "<IMAGENET_VAL_ROOT>" \
    --output_path     "<RESULTS_ROOT>/baseline/baseline_accuracy.json"
```

**3. Compression sweeps**

```bash
# Magnitude pruning (full grid)
python -m scripts.magnitude_pruning \
    --checkpoint_root "<CHECKPOINT_ROOT>" \
    --val_root        "<IMAGENET_VAL_ROOT>" \
    --output_path     "<RESULTS_ROOT>/pruning/magnitude.json"

# PE-buffer magnitude pruning (auxiliary)
python -m scripts.magnitude_pruning \
    --checkpoint_root "<CHECKPOINT_ROOT>" \
    --val_root        "<IMAGENET_VAL_ROOT>" \
    --output_path     "<RESULTS_ROOT>/pruning/magnitude_pe_buffer.json" \
    --scopes pe_buffer_cache

# Structured pruning (heads + neurons)
python -m scripts.structured_pruning \
    --checkpoint_root "<CHECKPOINT_ROOT>" \
    --val_root        "<IMAGENET_VAL_ROOT>" \
    --output_path     "<RESULTS_ROOT>/pruning/structured.json"

# Post-training quantization (full grid)
python -m scripts.ptq_quantization \
    --checkpoint_root "<CHECKPOINT_ROOT>" \
    --val_root        "<IMAGENET_VAL_ROOT>" \
    --output_path     "<RESULTS_ROOT>/quantization/ptq.json"

# PE-buffer PTQ (auxiliary)
python -m scripts.ptq_quantization \
    --checkpoint_root "<CHECKPOINT_ROOT>" \
    --val_root        "<IMAGENET_VAL_ROOT>" \
    --output_path     "<RESULTS_ROOT>/quantization/ptq_pe_buffer.json" \
    --scopes pe_buffer_cache
```

**4. Layer-wise CKA**

```bash
python -m scripts.cka_pruning \
    --checkpoint_root "<CHECKPOINT_ROOT>" \
    --val_root        "<IMAGENET_VAL_ROOT>" \
    --output_path     "<RESULTS_ROOT>/cka/cka_pruning.json"
```

**5. Aggregate analysis**

```bash
python -m scripts.analyze_results \
    --baseline   "<RESULTS_ROOT>/baseline/baseline_accuracy.json" \
    --magnitude  "<RESULTS_ROOT>/pruning/magnitude.json" \
    --structured "<RESULTS_ROOT>/pruning/structured.json" \
    --ptq        "<RESULTS_ROOT>/quantization/ptq.json" \
    --output_dir "<RESULTS_ROOT>/analysis"
```

**6. Regenerate paper figures and appendix tables**

The asset generators expect JSON inputs in a flat directory, so first
stage the experiment outputs:

```bash
mkdir -p "<RESULTS_ROOT>/_assets_staging"
cp "<RESULTS_ROOT>/baseline/baseline_accuracy.json" \
   "<RESULTS_ROOT>/pruning/magnitude.json" \
   "<RESULTS_ROOT>/pruning/magnitude_pe_buffer.json" \
   "<RESULTS_ROOT>/pruning/structured.json" \
   "<RESULTS_ROOT>/quantization/ptq.json" \
   "<RESULTS_ROOT>/quantization/ptq_pe_buffer.json" \
   "<RESULTS_ROOT>/cka/cka_pruning.json" \
   "<RESULTS_ROOT>/_assets_staging/"
```

Then generate the main-text figures (PDF/PNG):

```bash
python -m scripts.generate_results_assets \
    --json_dir "<RESULTS_ROOT>/_assets_staging" \
    --out_dir  "<RESULTS_ROOT>/results_assets"
```

And the 14 LaTeX appendix tables:

```bash
python -m scripts.generate_appendix_assets \
    --json_dir "<RESULTS_ROOT>/_assets_staging" \
    --out_dir  "<RESULTS_ROOT>/appendix_assets"
```

The notebook (`02_compression_main_workflow.ipynb` section 9) performs
this staging automatically.

### Recommended data layout

The pretrained checkpoints are large (~3.8 GB) but the compression
outputs are small — all JSONs together are under 5 MB. A natural layout
keeps the checkpoints outside the Git working tree:

```
<DATA_HOME>/                            # any location with sufficient disk space
└── Trained models_ImageNet100/         # 12 checkpoints (downloaded from Google drive)

<RESULTS_ROOT>/                         # all compression outputs (small, ~5 MB total)
├── baseline/
├── pruning/
├── quantization/
├── cka/
└── analysis/

vit-pe-compression/                     # this Git repository (small, code + results)
└── (code + JSON results + figures)
```

In the Colab notebooks `<DATA_HOME>` and `<RESULTS_ROOT>` are paths on
your Google Drive (defaults
`/content/drive/MyDrive/Trained models_ImageNet100` and
`/content/drive/MyDrive/pe_compression_experiment/results`). For local
execution they can be any directories.

---

## Notes on local execution

All scripts accept their input and output paths as CLI arguments, so no
code changes are needed to relocate the pipeline. A few practical points:

- **ImageNet-100 dataset structure** required by the compression scripts
  is the standard ImageFolder layout: one subdirectory per class, with
  image files inside. Pass the path to the parent of the class
  subdirectories as `--val_root`.

- **GPU memory.** Feature extraction is the only step that benefits from
  a high-memory GPU. Magnitude pruning, structured pruning, PTQ, and CKA
  computation are CPU-friendly. If you hit out-of-memory on your card,
  reduce `--batch_size` (default 128).

- **Wall-clock time.** The full compression pipeline (876 evaluations
  plus 96 CKA computations) completes in approximately 1.5 hours on a
  single NVIDIA RTX PRO 6000 Blackwell Server Edition (102 GB) with
  PyTorch 2.11 and CUDA 12.8. Pretraining the underlying twelve
  checkpoints required ~102 GPU-hours on a mixed pool of H100 and A100
  sessions.

---

## Key results

### Structured head pruning at the headline ratio (global scope, r=4/12)

The single most extreme cross-PE compression gap in the study occurs
under global head pruning when one third of the 144 attention heads
are removed:

| PE family | Top-1 at r=4/12 (mean ± std, 3 seeds) | Δ from baseline |
| --- | --- | --- |
| **ALiBi**   | **65.2 ± 1.2%** | −15.9pp |
| Sinusoidal  | 41.1 ± 5.8%     | −40.7pp |
| Learned     |  4.8 ± 1.3%     | −74.5pp |
| RoPE        |  2.5 ± 0.6%     | −82.0pp |

The 26× ratio between ALiBi (65.2%) and RoPE (2.5%) at the same
compression level is the strongest cross-PE separation in the study.
The ranking inverts the baseline accuracy ordering — RoPE achieves the
highest uncompressed top-1 (84.5%) but is the most fragile to global
head pruning. Full-ratio sweeps (r=1/12, 2/12, 6/12, 8/12) and per-layer
scope results are in paper Section 4.3 and Appendix A.2.

### Positional buffer compression (X-cross dichotomy)

| PE | r=0 (baseline) | INT2 PTQ (Δ from baseline) | r=0.9 magnitude (Δ from baseline) |
| --- | --- | --- | --- |
| RoPE       | 84.5 ± 0.3 | 83.6 ± 0.3 (**−0.9pp**) | 14.8 ± 1.5 (**−69.7pp**) |
| Sinusoidal | 81.8 ± 0.4 | 80.9 ± 0.3 (**−0.8pp**) | 5.5 ± 0.9 (**−76.2pp**) |
| ALiBi      | 81.1 ± 0.3 | 73.5 ± 0.4 (**−7.6pp**) | 73.5 ± 0.4 (**−7.5pp**) |
| Learned    | 79.3 ± 0.4 | 75.4 ± 0.7 (**−3.8pp**) | 77.8 ± 0.6 (**−1.5pp**) |

The two PE-buffer compression modalities partition the four families
into two storage-format classes with inverted robustness profiles
(paper Section 5.1).

### Layer-wise CKA (MLP-pruning depth gradient)

| Block | Mean CKA at r=0.7 | Notes |
| ---: | --- | --- |
|  0 | 0.99 ± 0.00 | Early representations preserved |
| **10** | **0.56 – 0.71** (minimum) | Maximum compression damage |
| 11 | 0.71 – 0.79 (recovery) | Classifier-head proximity |

The minimum sits at block 10 for all four PE families at r=0.7, with a
block-11 partial recovery; the four PE families differ in the magnitude
of CKA loss, not in its location. This is distinct from the block-9
representational divergence peak reported in the companion transfer
paper (paper Section 4.5).

Full per-layer tables, ratio sweeps, and the figure are produced by
`cka_pruning.py` and `generate_results_assets.py` respectively.

---

## Paper and citation

The paper is currently under submission. In the meantime, please
reference this repository directly if you build on the code or use the
results.

```bibtex
@misc{bandjur2026pecompression,
  title  = {Positional Encoding Determines Structural Redundancy in Vision Transformers},
  author = {Bandjur, Djoko and Bandjur, Milos and Micic, Aleksandar},
  year   = {2026},
  note   = {Manuscript under submission.
            Code: \url{https://github.com/djokobandjur/vit-pe-compression}}
}
```

---

## License

This repository uses a dual-licensing scheme that reflects the different
nature of its contents:

- **Source code** (all `.py` files, both notebooks under `notebooks/`)
  is released under the **MIT License** — see [`LICENSE`](LICENSE).

- **Result files and documentation** (the JSON files under `results/`,
  the PDF/PNG figures under `figures/`, this README, and the Zenodo
  deposit of this repository) are released under the **Creative Commons
  Attribution 4.0 International License** (CC BY 4.0). Full text:
  [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

- **Trained model checkpoints** (hosted on Google Drive; see *Trained models*
  above) are released under **CC BY 4.0 for research purposes**. The
  ImageNet-100 models are derivative artifacts of the ImageNet-1k dataset
  and remain subject to the
  [ImageNet terms of access](https://www.image-net.org/download.php) for
  any redistribution or commercial use.

- **The ImageNet-1k validation images** required to reproduce the
  ImageNet-100 experiments are governed by the
  [ImageNet terms of access](https://www.image-net.org/download.php) and
  are not redistributed in this repository.

If you use the code, cite the repository under MIT terms. If you use the
results, figures, or trained models in a derivative work, cite under
CC BY 4.0 terms (attribution required).
