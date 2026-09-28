# Final LBF experiment handoff

This folder contains only the material needed to understand, reproduce, and
extend the experiments that survived into the final dissertation.  It is not a
copy of the thesis workspace.  Old trials, abandoned pipeline names, paper
drafts, presentation material, and intermediate figures were intentionally
left out.

## What is here

- **Pipeline 1 (P1):** the 4,900-bit global pixel HBF.  Both the original
  165-filter layout and the corrected 167-filter thesis layout are explicit.
- **Pipeline 2 (P2):** the 48-landmark DCT LBF, with 12 retained bits per
  landmark (576 bits / 72 bytes).
- **Pipeline 3 (P3):** the 48-landmark ArcFace spatial LBF, with the fitted
  PCA-whitening/rotation reduced to one 512-by-12 linear encoder.
- The exact 9,742-image inventory, frozen identity partitions, membership
  roles, and enrollment/query use for every image.
- The final accepted template bits, thresholds, predictions, metrics,
  diagnostics, and parameter sweeps.
- Small tests that catch accidental protocol, layout, or result changes.

## The shortest possible reproduction

Create an environment with Python 3.11 or newer, install
`requirements-lock.txt`, and run:

```bash
python run.py verify
python run.py reproduce
python run.py figures
```

`verify` checks the bundled evidence without rebuilding Bloom filters.
`reproduce` rebuilds all 15 final HBF banks from the frozen template bits,
selects thresholds on development identities, freezes them, and evaluates the
held-out identities.  It writes to `reproduced/`; the accepted `results/`
folder is never overwritten.  `figures` redraws the final comparison and ROC
plots from the accepted CSV files.

The frozen bits are deliberate.  They make the primary result reproducible on
a laptop without redistributing face photographs, a 100+ MB landmark model, or
the ArcFace ONNX model.  To rerun from raw photographs, place the datasets and
models outside this folder, verify them against `frozen/model_checksums.toml`,
and use the readable frontend functions in `src/lbf_handoff/`.

## Dataset protocol

`data/selected_samples.csv` is the source of truth.  It contains one row per
photograph, never an absolute machine-specific path.  The important columns
are `partition`, `membership_role`, `protocol_use`, and
`preprocessing_success`.

- `fit` identities may fit preprocessing/frontends only.  They never become
  recognition queries.
- `development` identities choose integer score thresholds.
- `evaluation` identities are untouched until the final measurement.
- Faces94/95/96 use the successful first half of each enrolled identity's
  ordered photographs for enrollment; the remaining photographs are genuine
  probes.  Impostor identities are probe-only.
- FEI and FERET use the paired single-shot protocol: capture 0 enrolls and
  capture 1 is the genuine probe.  An impostor contributes capture 1 only.

The split is by identity, so no person appears in more than one partition.

## Pipeline notes

### P1: global pixels

The aligned face is masked, histogram equalized, reduced to 70-by-70, and
thresholded by the fit-set mean face.  `pipeline1.py` exposes:

- `mode="original"`: 153 complete 32-bit blocks, 11 complete 416-bit blocks,
  and one 4,900-bit block = **165 filters**.  This preserves the published
  omission of trailing partial blocks.
- `mode="thesis"`: 154 blocks at 32 bits, 12 blocks at 416 bits, and one top
  block = **167 filters**.  The final partial blocks are retained.  This is the
  dissertation baseline and the mode used by `reproduce`.

The handoff also records the legacy address routine used by the original code,
including its XOR fold and the unused tail produced for a non-power-of-two
Bloom size.
The final thesis evidence uses the domain-separated HMAC-SHA256/FNV-1a routine
in `bloom.py`; layout compatibility and hash compatibility are separate choices
so a future comparison cannot silently mix them.

### P2: landmark DCT

Each aligned landmark supplies a normalized 32-by-32 patch.  The patch is
reduced to 8-by-8, transformed by a 2-D DCT, and read in low-frequency zigzag
order after discarding the DC term.  Coefficients are thresholded by medians
fitted on fit identities only.  The final system retains the first 12 bits at
the 48 frozen landmarks.

### P3: ArcFace spatial descriptors

One 112-by-112 ArcFace-aligned image is passed through the frozen IResNet-50.
The named 512-by-7-by-7 intermediate tensor is bilinearly sampled at all
landmarks.  PCA whitening (512 to 64) and the seeded rotation are already
folded into the frozen 512-by-12 matrices.  A sign test gives 12 bits per
landmark; the same 48 landmarks as P2 are retained.

## Where to make changes

- Change a frontend in `pipeline1.py`, `pipeline2.py`, or `pipeline3.py`.
- Change only Bloom-filter behavior in `bloom.py`.
- Change a protocol only in `data.py`, then update the manifest and its tests.
- Add a diagnostic to `diagnostics.py` or a controlled sweep to `sweeps.py`.
- Keep final evaluation identities out of all fitting and selection code.

The comments are intentionally written for another researcher, not just for a
programmer.  Each non-obvious choice explains the experimental reason behind
it.
