# Dataset manifest

`selected_samples.csv` is the complete frozen inventory used by the final
thesis experiments.  It contains 9,742 photographs from 1,498 identities.
The photographs themselves are not redistributed here.

Columns:

- `dataset_index`: row address in that dataset's frozen template arrays.
- `identity`: dataset-native identity label.
- `capture_order`: numeric order used when assigning enrollment and probes.
- `relative_path`: path relative to the corresponding dataset root.
- `sha256`: checksum of the original photograph.
- `partition`: fit, development, or evaluation identity partition.
- `membership_role`: enrolled, impostor, or frontend-fit role.
- `protocol_use`: the exact use of this photograph, or why it was unused.
- `preprocessing_success` and `preprocessing_result`: the recorded detector and
  alignment outcome.

No absolute paths are stored, so the handoff can move between machines.

