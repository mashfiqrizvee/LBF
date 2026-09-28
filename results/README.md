# Accepted results

`final_metrics.csv` and `final_predictions.csv` are the accepted held-out
evaluation evidence.  The runner treats them as read-only references.

The `diagnostics/` directory contains only the analyses discussed in the final
thesis: hierarchy contribution, Bloom sizing, template information, bit
disagreement/collision behavior, and transition metrics.

The `sweeps/` directory contains the DCT bit-depth, landmark-count, region
ablation, and handcrafted-frontend comparisons that informed the final design.

The `figures/` directory is generated only from these CSV files.  It contains
no manually entered metric values.

