# Intervention Geometry Shapes Apparent Positional-Encoding Robustness in Vision Transformers

Reproducibility repository for the manuscript

**Intervention Geometry Shapes Apparent Positional-Encoding Robustness in Vision Transformers**  
Đoko Banđur and Miloš Banđur  
Faculty of Technical Sciences, University of Pristina, Kosovska Mitrovica, Serbia

> **Public replication release:** `v2.0.0` (2026-09-07).  
> The historical `v1.0.0` release remains preserved as provenance for the earlier compression-grid scope.

[![License: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Results: CC BY 4.0](https://img.shields.io/badge/results%20%26%20figures-CC%20BY%204.0-lightgrey.svg)](LICENSE_SCOPE.md)

## DOI policy

- **Concept DOI (all versions; resolves to the latest release):** `10.5281/zenodo.20527499`
- **Exact v2.0.0 version DOI:** `10.5281/zenodo.22556912`
- **Historical v1.0.0 version DOI:** `10.5281/zenodo.20527500`

The manuscript and Supplement cite the **concept DOI** so their persistent repository link always resolves to the newest public release. For exact byte-level citation or reproduction of v2.0.0, cite the **version DOI** `10.5281/zenodo.22556912`.

## What changed from v1.0.0?

The original repository captured the motivating ViT-B/16 compression grid: four positional-encoding families, three seeds, head/neuron pruning, magnitude pruning, post-training quantization, and layer-wise CKA. That historical code/results tree remains available in `scripts/`, `results/`, `models/`, and `notebooks/`.

The final study asks a stricter mechanistic question: **when can task damage under compression or ablation be interpreted as evidence of structural redundancy?** The v2.0.0 replication release therefore adds the controlled-intervention evidence chain used by the final paper:

- exact-count head masks and depth-topology controls;
- balanced `K=48` topology matching;
- separately frozen balanced-`K=72` selector replication;
- realized-operator and additive input-scale diagnostics;
- causal positional controls and scalar-displacement reversals;
- the explicitly **post-hoc** ALiBi LOW/MID/HIGH slope-permutation bracket;
- selector-held-out, random-repeat, DTCD, Taylor-omission, and Monte-Carlo sensitivity evidence;
- exact checkpoint identities, masks, calibration/sample manifests, protocol/run identifiers, and SHA-256 provenance.

## Authoritative v2.0.0 replication archive

The public replication archive is:

`VITPECOMP_REPLICATION_PACKAGE_v2_0_0_20260907.zip`

SHA-256:

`70c7ec84a30de6614c1ee7daa30f7eee41bfb278e678fca810f301608945556c`

This ZIP is deliberately **replication-only**. It does **not** contain the manuscript, Supplement, bibliography, cover letter, highlights, or Editorial Manager source package. It was assembled from byte-preserved scientific artifacts in the frozen source snapshots while excluding publication-development material.

Source provenance retained by hash:

- frozen scientific source snapshot `VITPECOMP_NN_SCIENTIFIC_FINAL_v0_20_20260902.zip` — SHA-256 `2c6e20e9f099cdd9aa0120cd231e29a9f31838b96e9626449a0b214157df3668`;
- governance source snapshot `VIT_PE_COMPRESSION_REVISION_CONTROL_v1_39_20260902.zip` — SHA-256 `1abd1279e7fecb8c2efe47c798e8436b201353d0ad8dd23ccafc73f98f1879b2`.

Those full source ZIPs are **not** the public v2.0.0 release because they include publication-development material. The public archive copies only the required code/notebooks, protocols, executed evidence, masks/manifests, registries, plot data, and expected outputs and carries its own release-level SHA-256 manifest.

See [`release/v2.0.0/RELEASE_NOTES.md`](release/v2.0.0/RELEASE_NOTES.md) and [`release/v2.0.0/SHA256SUMS.txt`](release/v2.0.0/SHA256SUMS.txt).

## Integrity status

The source scientific freeze recorded:

- 276/276 numeric verification checks: **PASS**;
- Monte-Carlo precision extension: **84/84 complete**;
- source scientific-master manifest: **210/210 SHA-256 PASS**;
- source governance v1.39 manifest: **501/501 SHA-256 PASS**.

The public v2.0.0 ZIP has a separate release manifest covering every distributed file.

## Checkpoints and dataset

Dataset images and trained checkpoint binaries are not duplicated in the public ZIP. Exact checkpoint SHA-256 identities and dataset/sample manifests are retained so external copies can be verified before execution.

The original twelve ViT-B/16 ImageNet-100 checkpoints used by the historical base-grid cohort remain available at the historical public storage location: [Google Drive folder (~3.8 GB)](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9).

That link is stated only for the original twelve-model cohort. Later frozen checkpoint extensions are identified by their hashes/manifests in v2.0.0 and are not implicitly claimed to be contained in that historical folder.

## Reproducibility map

The v2.0.0 archive contains:

```text
VITPECOMP_REPLICATION_PACKAGE_v2_0_0_20260907/
  governance/
  notebooks/
  protocols/
  executed_evidence/
  reproduction/
  expected_outputs/
  RELEASE_MANIFEST_SHA256.txt
```

The balanced-`K=72` selector result is a frozen same-estimand replication layer. The ALiBi LOW/MID/HIGH complete-derangement bracket is a designed **post-hoc diagnostic**, not a preregistered random sample from the assignment space. Those epistemic roles are preserved explicitly in the release records.

## Zenodo

Automatic GitHub-to-Zenodo release ingestion is disabled for this repository. Version `v2.0.0` is deposited manually as a new version of the existing Zenodo record so the exact replication ZIP above is the archived payload.

Citation metadata for the exact release are in [`CITATION.cff`](CITATION.cff). The manuscript-facing persistent link is the concept DOI `10.5281/zenodo.20527499`.

## Licensing

The repository contains mixed scholarly materials. The root MIT license applies to project code. Project-generated machine-readable result summaries and figures/tables are intended for reuse under CC BY 4.0. See [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) for the file-level scope.