# Checkpoint access and SHA-256 verification

The final study uses **24 author-trained ViT-B/16 ImageNet-100 checkpoints**: four positional-encoding families (Learned, Sinusoidal, RoPE, ALiBi) × six seeds (42, 123, 456, 789, 1011, 1213).

## Public checkpoint folder

https://drive.google.com/drive/folders/1QH3EG9mf6oSWwzwrhdp599S5VB_hShZx?usp=drive_link

The shared folder contains one subfolder per PE/seed combination, each with `best_model.pth` and `training_history.json`. At release audit, the folder exposed all 24 expected `best_model.pth` files for download.

## Byte-level identity check

On 2026-09-07, SHA-256 was recomputed over the 24 publicly shared `best_model.pth` files. The resulting family, seed, byte size, and SHA-256 values matched the frozen experiment manifests **24/24, with 0 mismatches**.

The frozen source manifests are:

- Discovery/base cohort (seeds 42/123/456): `executed_evidence/B0_EXACTCOUNT_HEAD_AUDIT_V2/EVIDENCE/B0_EXACTCOUNT_HEAD_AUDIT_V2/VITPECOMP_CHECKPOINT_MANIFEST_VITB_FP32_v1.csv`
- Held-out extension (seeds 789/1011/1213): `executed_evidence/D1_HELDOUT_N3_V1/VITPECOMP_D1_HELDOUT_CHECKPOINT_MANIFEST_v1.csv`

| PE | Seed | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| learned | 42 | 343559209 | `7fcca75916c2d6f0f64aa5c381812ad3a305ba1a04672e9288f4251ab683c536` |
| learned | 123 | 343559209 | `fbb8d70f72fb6ee1bb93b1d00cca663ffb222f489d16d8149f2e01efd65c351e` |
| learned | 456 | 343559209 | `214c157417326a6c452a51aed95e312a1eb709094595145e25c994ad05a0f3ee` |
| learned | 789 | 343559209 | `1ea21524c9da28c375071e41bd692a7f909328409ce56618dab4d8affabb93c6` |
| learned | 1011 | 343559209 | `353f600f5e5ae8fb87649b9f5c1c5a5b0fe1343f3236a994b5a9e270a20c05c5` |
| learned | 1213 | 343559209 | `d4ab6e7c51a86bccd4f988e1f70b6d098c28cc78f9be23db8e8cf6314f296c4e` |
| sinusoidal | 42 | 343559209 | `15a95418f02a736056c3f8a6fc9e6ee691f4f8305493a6ddc0e01094234bb66b` |
| sinusoidal | 123 | 343559209 | `730353da2ea1c25db6eaa77fee6c76de203ea7ffdb52ab425df7ef35096fd399` |
| sinusoidal | 456 | 343559209 | `7b85839b150bd0e77821c96e83569f51c25722031538fb43cbf8a15a81c0af71` |
| sinusoidal | 789 | 343559209 | `a11ca590daf022fee035245c697f79741a36b81a3cb34bd29ef4031cc57111e9` |
| sinusoidal | 1011 | 343559209 | `4837c33caa430b06fd92476fab8c35445f1e8e830117542bfed9f608a96de58b` |
| sinusoidal | 1213 | 343559209 | `2f4fe8bcb8884c44121dc75d68d4a0e59622b10f29253a84c4a0ac4cc518d03a` |
| rope | 42 | 343573260 | `8a37631ee2fb19662e24cfb0e16ab985f2cfacfb021d12c2399541909a528118` |
| rope | 123 | 343573260 | `1262c21428a304581eaa094db0c98db61b4d23c2a5d731d89faf50a4b9fc4631` |
| rope | 456 | 343573260 | `8d2a84e689c7d4f271259308ec12666a55654a3df19654ddfe27be13ecbb062d` |
| rope | 789 | 343573260 | `d30eeb09acc05c091cc22d760ccf2e33775524f2bd2d60baa50864212bd0a564` |
| rope | 1011 | 343573260 | `03187354a7de6bb52f2f44d6aa6b3354f568c4e17077cdbaca7dae2b8a1bd5e4` |
| rope | 1213 | 343573260 | `2f487321994592012f073ef1d93e2a49a4f3fda0b9e12da2b4a8744767ee6bde` |
| alibi | 42 | 344824960 | `6dc6e8290cc6510a0ba2035c0f1189ac1555ed19e22a71d635de2cc3821c11e7` |
| alibi | 123 | 344824960 | `59b0f4b9132c3995e0aaed5958e40a2e49298a4dc15f0fad8093a6e7b079d04f` |
| alibi | 456 | 344824960 | `d8bd63348aac6634b5b20f62bc3176184c1304d037e5bb7bf1fccbbe009746d4` |
| alibi | 789 | 344824960 | `f8d396596923cd6b13426994f2a3f93203aa37d45a5e459c45ea45b794a9187d` |
| alibi | 1011 | 344824960 | `991425061f058e8df5ee6995d027a10a0280c4c53be5ed8489617a814dff88de` |
| alibi | 1213 | 344824960 | `d66b7012323932579013607d86b83af3c97279110dd7c196b8578b7009965370` |

## Reviewer verification

After downloading the checkpoint folder, verify each file before running an experiment. For example, in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 ".\learned_seed42\best_model.pth"
```

or on Linux:

```bash
sha256sum 'learned_seed42/best_model.pth'
```

A checkpoint should be used only when its family/seed path, byte size, and SHA-256 match the table above or the corresponding frozen manifest.
