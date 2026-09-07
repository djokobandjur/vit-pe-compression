# v2.0.0 — public replication release

**Release date:** 2026-09-07  
**Paper:** *Intervention Geometry Shapes Apparent Positional-Encoding Robustness in Vision Transformers*  
**Authors:** Đoko Banđur and Miloš Banđur  
**Exact version DOI:** `10.5281/zenodo.22556912`  
**Concept DOI:** `10.5281/zenodo.20527499`

## DOI roles

The manuscript and Supplement cite the **concept DOI** `10.5281/zenodo.20527499`, which resolves to the latest version of the Zenodo record. Exact archival citation of this v2.0.0 payload uses the **version DOI** `10.5281/zenodo.22556912`.

The historical v1.0.0 version DOI is `10.5281/zenodo.20527500`.

## Public replication archive

Use the following byte-identical archive as the GitHub v2.0.0 release asset and the manually curated Zenodo v2.0.0 payload:

`VITPECOMP_REPLICATION_PACKAGE_v2_0_0_20260907.zip`

SHA-256:

`c93e8f3625c9b784458ea647e1756f66a8238a23830e1c94267c8215f116d119`

The archive contains replication materials only: code/notebooks, frozen protocol locks, masks/manifests, executed raw/aggregated evidence, sensitivity analyses, checkpoint/sample provenance, plot data, expected outputs, and a release-level SHA-256 manifest.

It deliberately excludes the manuscript, Supplement, bibliography, cover letter, highlights, Editorial Manager source package, and manuscript-development/release directories.

## Source provenance

The retained artifacts were copied byte-for-byte from two frozen source snapshots:

- `VITPECOMP_NN_SCIENTIFIC_FINAL_v0_20_20260902.zip` — SHA-256 `2c6e20e9f099cdd9aa0120cd231e29a9f31838b96e9626449a0b214157df3668`;
- `VIT_PE_COMPRESSION_REVISION_CONTROL_v1_39_20260902.zip` — SHA-256 `1abd1279e7fecb8c2efe47c798e8436b201353d0ad8dd23ccafc73f98f1879b2`.

Those complete source ZIPs are not distributed as v2.0.0 because they also contain publication-development files outside the public replication boundary.

## Scientific-release status

The source freeze records 276/276 numeric checks PASS, 84/84 Monte-Carlo precision-extension evaluations complete, 210/210 scientific-master manifest entries verified, and 501/501 governance v1.39 entries verified. No scientific experiment is reopened by this release.

## Checkpoint availability closure

All 24 ViT-B/16 ImageNet-100 checkpoints used in the final study are publicly available in the shared Google Drive folder:

`https://drive.google.com/drive/folders/1QH3EG9mf6oSWwzwrhdp599S5VB_hShZx?usp=drive_link`

A release-time audit recomputed SHA-256 over every public `best_model.pth`. The public copies matched the frozen discovery/base and held-out checkpoint manifests **24/24**, with zero mismatches in PE family, seed, byte size, or SHA-256. The complete verification table is in `CHECKPOINT_ACCESS_AND_SHA256.md`.

## Main controlled results represented in v2.0.0

- exact-count/depth-topology control at balanced `K=48`;
- frozen balanced-`K=72` selector replication;
- realized-operator and additive input-scale audits;
- scalar physical/functional displacement diagnostics and within-family reversals;
- post-hoc invariant-preserving ALiBi LOW/MID/HIGH slope-permutation bracket;
- selector-held-out, random-repeat, DTCD, Taylor-omission, and Monte-Carlo sensitivity analyses;
- exact checkpoint/sample/calibration identities, masks, and SHA-256 provenance.

## Epistemic-role note

The balanced-`K=72` selector replication is a separately frozen same-estimand replication layer. The ALiBi LOW/MID/HIGH complete-derangement bracket is explicitly **post-hoc** and should not be described as preregistered or as a random sample of the complete slope-assignment space.

## Zenodo workflow

Automatic GitHub-to-Zenodo synchronization is disabled. The exact ZIP above is uploaded manually as a new version of the existing Zenodo record, using reserved version DOI `10.5281/zenodo.22556912`. The concept DOI `10.5281/zenodo.20527499` remains the stable manuscript-facing resolver to the latest version.