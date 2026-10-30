# GitHub implementation comparison

Searches included `RoadSignNet`, `GTSRB tensorflow keras traffic sign classification license:mit`, and `traffic sign recognition keras GUI license:mit`. Four candidates were cloned and inspected. No exact end-to-end match was verified; a matching name alone does not establish a matching implementation.

| Repository | Evidence inspected | Match and differences |
|---|---|---|
| [Vishal-kr6/traffic-signs-classification-cnn](https://github.com/Vishal-kr6/traffic-signs-classification-cnn) | Python script, README, MIT license | **Selected baseline:** 32×32 RGB, Keras, GTSRB, 43 outputs, 32/64 convolution blocks. Missing upload UI, saved bundle, official held-out evaluation, track-aware validation and per-class metrics. |
| [manasak2010/RoadSignNet](https://github.com/manasak2010/RoadSignNet) | README, repository file listing | Same name and broadly similar stack, but describes video/real-time detection and clustering; no license file was found in the cloned revision. Not selected for copying. |
| [seta-hosseini/gtsrb-traffic-sign-classification](https://github.com/seta-hosseini/gtsrb-traffic-sign-classification) | README, MIT license, file listing | GTSRB and Keras, but 64×64 input and multiple ANN/CNN/transfer-learning experiments. Less direct match to this report. |
| [Muhammad-AI-K/Traffic-Sign-Detector](https://github.com/Muhammad-AI-K/Traffic-Sign-Detector) | README, MIT license, file listing | GTSRB/Keras classifier, but 50×50 inputs and a different model configuration. |

Upstream accuracy figures were not independently reproduced and are not used as RoadSignNet results. The selected baseline is extended to implement the report; it is not claimed to be an exact original match.
