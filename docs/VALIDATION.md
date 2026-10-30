# Validation evidence

Verified in this session on **30 October 2026**, Python 3.12.14, Linux x86_64, CPU-only TensorFlow 2.20.0 (`tensorflow-cpu` distribution), Keras 3.15.1, NumPy 2.2.6, OpenCV 4.12.0.88, Streamlit 1.50.0.

## Executed checks

- `python -m ruff check .` — **passed**.
- `python -m pytest -q` — **17 passed in 7.84 seconds**.
- Original MIT license and archived upstream Python source — verified byte-for-byte against the pulled repository.

The test run emitted eight upstream Keras/NumPy deprecation warnings concerning the array `copy` keyword. They did not prevent training, serialization, inference or evaluation.

## Coverage against report test cases

| Report ID | Executed validation | Result |
|---|---|---|
| T01 | Official PPM layout and Kaggle-style class folders | Pass with generated fixtures |
| T02 | All 43 class IDs, including Stop and directional labels | Pass |
| T03 | RGB/grayscale decoding, RGB channel order, shape, dtype and normalization | Pass |
| T04 | One CPU training epoch with validation and augmentation | Pass on synthetic data |
| T05 | Best checkpoint save/reload, stable predictions, checksum and mapping errors | Pass |
| T06 | Inference, top-three ranking and UI prediction rendering | Pass on synthetic input; real-sign accuracy pending |
| T07 | Invalid image, unsupported format and in-app error message | Pass |
| T08 | Softmax shape/range/sum and invalid output handling | Pass |
| T09 | Confusion matrix CSV/PNG and per-class metric exports | Pass on synthetic held-out fixtures |
| T10 | Training and prediction with GPU disabled | Pass |

Additional checks cover reproducible track-disjoint splits, exact-byte overlap detection, refusal of test annotations pointing at training images, safe prediction CSV logging, missing-model UI, and generated prediction downloads.

The integration fixture contains **258 random training-pool images** over 43 labels and three tracks per label. The split is **172 training / 86 validation**, with **43 additional held-out synthetic images**. These are random test fixtures, not traffic-sign photographs. No fixture model is distributed as a usable sign recognizer. UI inference tests provide an in-memory upload because Streamlit AppTest has no native upload setter; they execute the actual app's prediction/error branches.

## Pending real-data experiment

The attached report provided architecture and a test plan, but no dataset or trained weights. The managed environment's allowed-host policy does not include the official GTSRB or Kaggle dataset hosts. Consequently:

- Full GTSRB training has **not** been performed.
- Real GTSRB test accuracy, macro precision/recall/F1 and benchmark timing are **not measured**.
- No production model, real-sign demonstration screenshot or benchmark confusion matrix is claimed.

Run the README training and evaluation commands with the official dataset to produce those artifacts. Do not use the synthetic test metrics as report results. Hardware-dependent model performance and real-world generalization remain unverified.
