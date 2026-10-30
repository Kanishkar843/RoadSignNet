# Report-to-code mapping

Reference: uploaded *Kanishkar_B_RoadSignNet_DLCV_Report.pdf*, “RoadSignNet: Traffic Sign Classification Using Deep Learning and Computer Vision”, Kanishkar B. The report is used as a specification and is not redistributed in this public repository.

| Report requirement | Implementation |
|---|---|
| §2.3 / §3.2.2: 32×32 RGB, values 0–1 | `roadsignnet/preprocess.py`; same function in data loading and inference |
| §3.2.3: compact 32/64-filter CNN, pooling, 43 softmax outputs | `roadsignnet/model.py` |
| §3.3.2 / §5.3.3: Adam, sparse categorical cross-entropy | `roadsignnet/model.py`, `roadsignnet/train.py` |
| §5.3.1: GTSRB class IDs and official test separation | `roadsignnet/data.py`, independent evaluation command |
| §2.1.4: cautious augmentation, no direction-changing flips | Training-only Keras augmentation layers |
| §3.2.4 / §5.3.4: saved model, label, score, top three | `roadsignnet/artifacts.py`, `roadsignnet/predict.py`, `app.py` |
| §3.3.3: versioned mapping/configuration and optional CSV | Checksummed bundle metadata, split manifest, `--log` |
| §5.3.5: per-class metrics and confusion matrix | `roadsignnet/evaluate.py` |
| §6.1: functional test cases | `tests/` and `docs/VALIDATION.md` |
| §6.2: actual outputs and measured results | Training/evaluation commands generate real experiment artifacts; benchmark values remain unmeasured |
| §6.3: educational classification only | UI and README explain scope and score limitations |

## Explicit limits

The report asks for images without a visible sign to be rejected or flagged as uncertain. A closed-set softmax CNN cannot reliably determine whether a sign is present. The interface therefore requires review of **all** predictions, adds a low-confidence warning below a configurable threshold, and explains that high scores do not reject unknown objects. No unimplemented sign-presence detector is claimed.

The report's mock-up and example 0.91 score are illustrative. Neither that score nor any accuracy number was copied into this application's results. Full GTSRB training, held-out metrics and real-sign screenshots remain an experiment to execute with the dataset.
