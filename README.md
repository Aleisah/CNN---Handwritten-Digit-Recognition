# Handwritten Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) in MATLAB to recognize handwritten digits from a dataset of 28x28 grayscale images. The model classifies digits into 10 categories, representing digits 0 through 9. This project leverages MATLAB's deep learning tools to create an efficient and accurate model for digit classification.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Usage Instructions](#usage-instructions)
- [Key Takeaways](#key-takeaways)
- [License](#license)

## Overview

The goal of this project is to build a CNN model that can accurately classify handwritten digits. The model is trained on a dataset of 28x28 grayscale images and is evaluated using accuracy and loss metrics. The model achieves high performance, making it suitable for real-world digit recognition applications.

## Dataset

The dataset used for this project is MATLAB’s built-in `DigitDataset`, which contains images of handwritten digits organized into folders labeled with the corresponding digit. The dataset is split as follows:
- **Training and Validation:** 80% of the dataset
- **Testing:** 20% of the dataset

The dataset is loaded using MATLAB's `imageDatastore` function, which simplifies data loading and preprocessing.

## Methodology

### Dataset Preparation:
1. The dataset is loaded using the `imageDatastore` function.
2. It is split into training (80%) and testing (20%) sets.
3. The training set is further divided into 80% training and 20% validation sets.

### CNN Architecture:
1. **Input Layer:** Accepts 28x28 grayscale images.
2. **Convolutional Layers:** Two convolutional layers with 3x3 filters (16 and 32 filters).
3. **Batch Normalization & ReLU:** Normalize activations and introduce non-linearity.
4. **Max Pooling:** Reduce spatial dimensions by half.
5. **Fully Connected Layers:** 128 neurons in the first fully connected layer, followed by 10 neurons (one for each digit).
6. **Dropout Layer:** Dropout rate of 50% to prevent overfitting.
7. **Softmax & Classification Layers:** Convert scores to probabilities and assign class labels.

### Training:
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Epochs:** 10
- **Mini-Batch Size:** 128
- **Validation Data:** Used to monitor performance during training.
- **Early Stopping:** Monitoring of validation performance every 30 iterations.

### Testing & Evaluation:
- After training, the model is evaluated on the test dataset.
- Performance metrics such as accuracy and loss are calculated.
- A single test image is displayed with predicted and actual labels.

## Code Explanation

The project contains the following key components:

1. **Training Script (`train_model.m`)**
   - Loads and splits the dataset.
   - Defines the CNN architecture.
   - Trains the model using the `trainNetwork` function.
   - Saves the trained model for later use.

2. **Testing Script (`test_model.m`)**
   - Loads the trained model.
   - Evaluates the model on the test set.
   - Displays a random test image with its predicted label.

3. **Random Predictions Script (`random_predictions.m`)**
   - Loads the trained model.
   - Makes predictions on a random set of test images.
   - Displays the predictions in a 5x5 grid.

## Results

During training, the model achieved the following:
- **Validation Accuracy:** 99.81% after 10 epochs.
- **Loss:** Both training and validation loss consistently decreased, indicating effective learning.

### Test Accuracy:
The model performed well on the test dataset, with an overall high accuracy, demonstrating the robustness and generalization of the model.

### Visualizations:
Random test images were classified with high accuracy. Check the "Random Predictions" section for visual examples.

## Usage Instructions

1. **Clone this Repository:**
   ```bash
   git clone https://github.com/yourusername/handwritten-digit-recognition.git
