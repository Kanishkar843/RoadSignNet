# Traffic Signs Classification with CNNs

This project implements a Convolutional Neural Network (CNN) in Keras/TensorFlow to classify traffic signs using the GTSRB dataset.

## Features
- Data loading and preprocessing (resize, normalization, one-hot encoding)
- CNN architecture for multi-class classification
- Model training and evaluation
- Visualization of training progress and predictions

## Dataset
- [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Download and unzip the GTSRB dataset into the `data/` folder.

3. Run the script:
    ```bash
    python traffic_signs_cnn.py
    ```

4. Check the `images/` folder for training curves and sample predictions.

## Requirements
- Python 3.8+
- TensorFlow / Keras
- numpy
- pandas
- matplotlib
- scikit-learn
- pillow

## Results
Sample output plots and prediction examples can be found in the `images/` folder.

---

Feel free to use and adapt this project!
