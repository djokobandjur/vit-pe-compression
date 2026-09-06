# Intervention Geometry Shapes Apparent Positional-Encoding Robustness in Vision Transformers

Reproducibility repository for the manuscript

**Intervention Geometry Shapes Apparent Positional-Encoding Robustness in Vision Transformers**  
Đoko Banđur and Miloš Banđur  
Faculty of Technical Sciences, University of Pristina, Kosovska Mitrovica, Serbia

> **Public replication release:** `v2.0.0` (2026-09-07).  
> The historical `v1.0.0` release is preserved for provenance, but it represents the earlier compression-grid manuscript and is **not** the authoritative replication release for the final study.

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

## Authoritative v2.0.0 replication archive

The public scientific replication archive is:

`VITPECOMP_PUBLIC_REPLICATION_v2_0_0_20260907.zip`

SHA-256:

`aba6bb687262f1795f4614fbae6dc7637232a5cd8c0b00a6ff29dc800923e5a8`

This archive is deliberately **science-only** at the release boundary: it contains the byte-frozen scientific master plus release/citation/license metadata, but does not embed the later editorial Main/Supplement or Editorial Manager source bundles. That separation keeps the scientific archive immutable and avoids a DOI-insertion circularity in the submission source.

The core payload is:

`VITPECOMP_NN_SCIENTIFIC_FINAL_v0_20_20260902.zip`

SHA-256:

`2c6e20e9f099cdd9aa0120cd231e29a9f31838b96e9626449a0b214157df3668`

The `NN` token in this frozen inner filename is a historical artifact identifier and has not been renamed because its filename and hash are part of the provenance lock.

The scientific master contains the complete governance archive `VIT_PE_COMPRESSION_REVISION_CONTROL_v1_39_20260902.zip` and the executed evidence/protocol packages.

See [`release/v2.0.0/RELEASE_NOTES.md`](release/v2.0.0/RELEASE_NOTES.md) and [`release/v2.0.0/SHA256SUMS.txt`](release/v2.0.0/SHA256SUMS.txt).

## Integrity status of the frozen scientific master

At scientific closure:

- 276/276 numeric verification checks: **PASS**;
- Monte-Carlo precision extension: **84/84 complete**;
- scientific-master manifest: **210/210 SHA-256 PASS**;
- governance v1.39 manifest: **501/501 SHA-256 PASS**;
- post-ZIP verification: **0 missing / 0 extra / 0 hash mismatch**;
- no new scientific experiment was authorized after the freeze.

## Repository layout

The Git tree retains the original base-grid implementation:

```text
vit-pe-compression/
├── data/                 # ImageNet-100 class/validation metadata helpers
├── models/               # ViT-B/16 + Learned/Sinusoidal/RoPE/ALiBi implementations
├── notebooks/            # historical base-grid workflow/sanity check
├── results/              # original 876-configuration compression grid
├── scripts/              # pruning, PTQ, CKA, analysis and asset generation
├── release/v2.0.0/       # final-release notes and immutable payload hash
├── CITATION.cff
├── .zenodo.json
├── LICENSE
└── LICENSE_SCOPE.md
```

The **later controlled-intervention program is distributed in the v2.0.0 replication archive**, rather than being retroactively mixed into the historical base-grid directories.

## Checkpoints and dataset

The original base-grid layer uses twelve author-trained ViT-B/16 ImageNet-100 checkpoints (Learned, Sinusoidal, RoPE, ALiBi × seeds 42/123/456); later replication stages use the explicitly documented frozen checkpoint extensions recorded in the scientific master.

Checkpoint binaries and ImageNet images are not duplicated in the release because of size and distribution constraints. The frozen scientific master retains exact checkpoint SHA-256 identities and dataset/sample manifests so externally stored copies can be verified before execution. The original twelve ViT-B checkpoints remain available from the public storage location documented in the historical workflow.

## Reproducibility map

For the final paper, start from the v2.0.0 public replication archive rather than from the historical notebooks alone. The scientific master contains the protocols, executed evidence, sensitivity analyses, hashes, scientific manuscript master, and governance provenance needed to trace the final results.

The balanced-`K=72` selector result is a frozen same-estimand replication layer. The ALiBi LOW/MID/HIGH complete-derangement bracket is a designed **post-hoc diagnostic**, not a preregistered random sample from the assignment space. Those epistemic roles are preserved explicitly in the manuscript and in the release records.

## Zenodo / citation

The repository already has a Zenodo record associated with the historical `v1.0.0` release (`10.5281/zenodo.20527499`). The final study should cite the **new v2.0.0 version DOI**, not the historical release DOI.

For v2.0.0, the complete replication ZIP is deposited as a **manual new version** of the existing Zenodo record. This is intentional: Zenodo's GitHub integration archives the repository source snapshot, but extra GitHub release assets are not automatically included. The v2.0.0 version DOI will be added here and to `CITATION.cff` once reserved/published.

Citation metadata are prepared in [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).

## Licensing

The repository contains mixed scholarly materials. The root MIT license applies to project code. Project-generated machine-readable result summaries and figures/tables are intended for reuse under CC BY 4.0. Manuscript text/source is included for exact scholarly provenance and is not covered by the MIT software license. See [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) for the complete scope statement.
