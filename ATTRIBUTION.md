# Attribution and implementation history

## Reused source

- Upstream: https://github.com/Vishal-kr6/traffic-signs-classification-cnn
- Author/copyright holder: **Vishal Kumar**, 2025
- Inspected and pulled revision: `2bab8902054d8eaa2585080bae2c421703f8f8f5`
- License: MIT, retained verbatim in `LICENSE` and `third_party/upstream/LICENSE`.
- Original Python script and README: preserved verbatim in `third_party/upstream/` for review. These are historical references, not the maintained execution entrypoints.
- The compact 32/64-convolution, dense-128 CNN architecture and GTSRB workflow were adapted from this baseline.

## Changes prepared for this project

The following implementation work was generated with AI assistance for **Kanishkar B**, aligned to the supplied RoadSignNet report:

- Modular package and separate train, evaluate and predict commands.
- Shared RGB/OpenCV preprocessing and input validation.
- Official and Kaggle GTSRB layout readers.
- Per-class sign-track validation split and byte-hash leakage checks.
- Sparse-label training, semantic-preserving augmentation, early stopping and checkpoint selection.
- Model/label/preprocessing bundle metadata and checksums.
- Held-out per-class metrics, confusion matrix, prediction tables and latency measurement.
- Streamlit image-upload UI, top-three scores, confidence review and JSON download.
- Reproducible dependency versions, CPU smoke tests, CI workflow and documentation.

Kanishkar B is the project owner and report author. This attribution does not claim that the upstream source was written by Kanishkar B, nor that benchmark experiments have been completed. No upstream plots or accuracy claims are presented as results of this adaptation.

## Data and dependencies

GTSRB belongs to its original dataset authors and has its own terms. Download it from the official project or a documented mirror and retain the original labels and split. Dataset files are not redistributed here. Python dependencies retain their respective licenses.
