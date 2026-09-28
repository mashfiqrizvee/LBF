# Frozen artifacts

- `templates.npz` stores packed final P1/P2/P3 bits for every manifest row.
  This is the input to the exact result reproduction.
- `selected_landmarks.json` freezes the common 48-landmark P2/P3 subset.
- `faces_encoder.npz` is the P3 fit-only PCA/rotation encoder for the three
  Essex datasets.
- `fei_feret_encoder.npz` is the corresponding fit-only encoder for FEI/FERET.
- `thresholds.json` records development-selected thresholds frozen before the
  evaluation pass.
- `model_checksums.toml` identifies large external landmark and ArcFace model
  files needed only when regenerating templates from raw photographs.

The `.npz` encoders contain `mean` (512 values), `matrix` (512 by 12), the 64
PCA explained variances, and the fit-image count.  Runtime P3 encoding requires
only `mean` and `matrix`.

