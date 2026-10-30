# Dataset setup

Use the [official German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html). Obtain the **final training images**, **final test images**, and **final test ground-truth labels**. The benchmark has 39,209 official training images and 12,630 test images across 43 classes.

Review and follow the dataset's download/access terms. Extract the archives locally. Do not merge the test images into the training folders. Keep the original filenames: their track identifiers are needed to prevent neighboring images of one physical sign appearing in both training and validation.

## Official layout

```text
data/
├── GTSRB/
│   ├── Final_Training/Images/
│   │   ├── 00000/00000_00000.ppm
│   │   ├── 00001/...
│   │   └── 00042/...
│   └── Final_Test/Images/00000.ppm
└── GT-final_test.csv
```

The test CSV must contain `Filename` and `ClassId`; semicolon-separated official files are supported. Supply `--data-dir data --test-csv data/GT-final_test.csv`. The final-training `Images` directory itself is also accepted by the training command.

## Kaggle repack layout

The commonly used [GTSRB Kaggle repack](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) can also be used after checking its provenance and split:

```text
data/
├── Train/
│   ├── 0/00000_00000_00000.png
│   ├── 1/...
│   └── 42/...
├── Test/00000.png
└── Test.csv
```

Supply `--data-dir data --test-csv data/Test.csv`. `Path` and `ClassId` columns are required for this layout. Test labels are read only by evaluation.

## Split policy

Training groups are `(class ID, track ID)`. Within each class, 20% of tracks (rounded, with at least one track per partition) are held out for validation using seed 42. This preserves all 43 categories and keeps related frames together. At least two tracks per class are required. The actual validation image fraction can differ from 20% because complete tracks are selected.

All images are decoded and validated before training. Cross-partition exact-byte duplicates are rejected. This audit cannot detect every transformed/near-duplicate photograph or certify a mirror's provenance; use the official split. The immutable manifest is saved alongside the checkpoint; evaluation rejects test images that match training/validation image hashes.

## Resource notes

The loader holds resized float32 images in memory (approximately 460 MB for the official training set, plus framework and training allocations). Use an 8 GB RAM machine. Start with one epoch to verify setup, then use a separate empty output directory for the full experiment. Full training duration depends on hardware.

The attached report did not include the dataset or trained weights. This session's network policy did not allow the official dataset or Kaggle hosts, so benchmark training was not performed here. The source code and commands are ready for local or cloud-notebook training with the actual data.
