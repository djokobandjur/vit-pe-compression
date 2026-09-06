# Intervention Geometry Shapes Apparent Positional-Encoding Robustness in Vision Transformers

Reproducibility repository for the manuscript

**Intervention Geometry Shapes Apparent Positional-Encoding Robustness in Vision Transformers**  
Đoko Banđur and Miloš Banđur  
Faculty of Technical Sciences, University of Pristina, Kosovska Mitrovica, Serbia

> **Submission release:** `v2.0.0` (prepared 2026-09-06).  
> The historical `v1.0.0` release is preserved for provenance, but it represents the earlier compression-grid manuscript and is **not** the authoritative replication release for the final Neural Networks submission.

[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Results: CC BY 4.0](https://img.shields.io/badge/results%20%26%20figures-CC%20BY%204.0-lightgrey.svg)](LICENSE_SCOPE.md)

## What changed from v1.0.0?

The original repository captured the motivating ViT-B/16 compression grid: four positional-encoding (PE) families, three seeds, head/neuron pruning, magnitude pruning, post-training quantization, and layer-wise CKA. That grid remains useful and is preserved in `scripts/`, `results/`, `models/`, and `notebooks/`.

The final paper asks a stricter mechanistic question: **when can task damage under compression or ablation be interpreted as evidence of structural redundancy?** The answer requires auditing the *realized intervention*, not only its nominal sparsity, head count, or bit width.

The `v2.0.0` release therefore adds the complete frozen evidence chain used by the final manuscript:

- exact-count head masks and depth-topology controls;
- balanced `K=48` comparison showing that block balancing cuts the 62.55-point ALiBi–RoPE head-masking gap by 61.55 points across 3/3 seed pairs, while the residual balanced gap remains unresolved;
- a separately frozen balanced-`K=72` selector replication in which activation ranking beats random for ALiBi and loses for RoPE;
- realized-operator diagnostics for additive PE, RoPE, and ALiBi;
- the additive input-scale audit (including the canonical Sinusoidal `sqrt(d/2)` scale effect);
- scalar displacement analyses and within-family order reversals;
- the **post-hoc** ALiBi LOW/MID/HIGH slope-permutation bracket, which preserves the clean slope multiset and registered norm invariants while damage spans 29.63 to 76.39 percentage points with LOW<MID<HIGH in 6/6 checkpoints;
- selector-held-out, random-repeat, DTCD, Taylor-omission, and Monte-Carlo sensitivity records;
- exact checkpoint identities, sample/calibration manifests, protocol/run identifiers, source hashes, and artifact SHA-256 provenance.

## Authoritative v2.0.0 release asset

The final public replication payload is packaged as:

`VITPECOMP_NN_PUBLIC_REPLICATION_v2_0_0_20260906.zip`

SHA-256:

`00ca5b071de16c5c5930ea1901c02e3e0dc86a32a64d9bb776a1b4b3498d5d6e`

The release asset contains three explicitly separated layers:

1. **Frozen scientific master** — `VITPECOMP_NN_SCIENTIFIC_FINAL_v0_20_20260902.zip`  
   SHA-256 `2c6e20e9f099cdd9aa0120cd231e29a9f31838b96e9626449a0b214157df3668`.
2. **Final Neural Networks submission/presentation layer** — `VITPECOMP_NN_MAIN_SUPPLEMENT_v0_28_4_LATEST_20260906.zip`  
   SHA-256 `52ef4c59c39895688278019c9cd1a52ad3f42a0fcad1160425987898428eac86`.
3. **Minimal Editorial Manager source package** — `VITPECOMP_NN_EM_SOURCE_PACKAGE_v0_28_4_20260906.zip`  
   SHA-256 `b76f2f16c0825b7dbf49d7e4a015639cf0fa2a85ff8e9a33188b78013561f6e0`.

The scientific master itself contains the complete governance archive `VIT_PE_COMPRESSION_REVISION_CONTROL_v1_39_20260902.zip` and the executed evidence/protocol packages. Keeping the scientific master and the final presentation layer separate preserves the scientific freeze while making the exact submitted source recoverable.

See [`release/v2.0.0/RELEASE_NOTES.md`](release/v2.0.0/RELEASE_NOTES.md) and [`release/v2.0.0/SHA256SUMS.txt`](release/v2.0.0/SHA256SUMS.txt).

## Integrity status of the frozen scientific master

At scientific closure:

- 276/276 numeric verification checks: **PASS**;
- Monte-Carlo precision extension: **84/84 complete**;
- scientific-master manifest: **210/210 SHA-256 PASS**;
- governance v1.39 manifest: **501/501 SHA-256 PASS**;
- post-ZIP verification: **0 missing / 0 extra / 0 hash mismatch**;
- no new scientific experiment was authorized after the freeze.

The later `v0.28.x` files are editorial/presentation layers only; they do not reopen the frozen scientific estimands.

## Repository layout

The Git tree retains the original base-grid implementation:

```text
vit-pe-compression/
├── data/                 # ImageNet-100 class/validation metadata helpers
├── models/               # ViT-B/16 + Learned/Sinusoidal/RoPE/ALiBi implementations
├── notebooks/            # legacy base-grid workflow/sanity check
├── results/              # original 876-configuration compression grid
├── scripts/              # pruning, PTQ, CKA, analysis and asset generation
├── release/v2.0.0/       # final-release notes and immutable payload hashes
├── CITATION.cff
├── .zenodo.json
├── LICENSE
└── LICENSE_SCOPE.md
```

The **later controlled-intervention program is distributed in the v2.0.0 release asset**, rather than being retroactively mixed into the historical base-grid directories.

## Checkpoints and dataset

The study uses twelve author-trained ViT-B/16 ImageNet-100 checkpoints (Learned, Sinusoidal, RoPE, ALiBi × seeds 42/123/456) for the original discovery/base-grid layer, plus the explicitly documented frozen checkpoint extensions used by later replication stages.

Checkpoint binaries and ImageNet images are not duplicated in the release because of size and distribution constraints. The frozen scientific master retains exact checkpoint SHA-256 identities and dataset/sample manifests so externally stored copies can be verified before execution. The original twelve ViT-B checkpoints remain available from the public storage location documented in the historical workflow.

## Reproducibility map

For the final paper, start from the v2.0.0 release asset rather than from the legacy notebooks alone:

- `01_SCIENTIFIC_MASTER/` — protocols, executed evidence, sensitivity analyses, hashes, scientific manuscript master and governance provenance;
- `02_FINAL_SUBMISSION_LAYER/` — final Main/Supplement, final figures/tables and presentation/invariant checks;
- `03_EM_SOURCE/` — minimal self-contained LaTeX source trees used for Editorial Manager compilation.

The balanced-`K=72` selector result is a frozen same-estimand replication layer. The ALiBi LOW/MID/HIGH complete-derangement bracket is a designed **post-hoc diagnostic**, not a preregistered random sample from the assignment space. Those epistemic roles are preserved explicitly in the manuscript and in the release records.

## Zenodo / citation

The repository already has a Zenodo record associated with the historical `v1.0.0` release (`10.5281/zenodo.20527499`). The final manuscript should cite the **new v2.0.0 Zenodo version DOI**, not the historical release DOI. That version DOI will be inserted here, in `CITATION.cff`, and in the manuscript Data and code availability statement immediately after the v2.0.0 deposit is published.

Citation metadata for the new release are prepared in [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).

## Licensing

The repository contains mixed scholarly materials. The root MIT license applies to project code. Project-generated machine-readable result summaries and figures/tables are intended for reuse under CC BY 4.0. Manuscript text/source is included for exact scholarly provenance and is not covered by the MIT software license. See [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) for the complete scope statement.
