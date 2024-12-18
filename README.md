# Handwritten Digit Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** in **MATLAB** for recognizing handwritten digits. The model classifies digits from **0 to 9** using a dataset of **28x28 grayscale images**.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Testing and Evaluation](#testing-and-evaluation)
- [Code Overview](#code-overview)
- [Results](#results)
- [Key Takeaways](#key-takeaways)

## Project Overview

The primary goal of this project is to develop a CNN capable of classifying handwritten digits from images. Using MATLAB's deep learning framework, we designed, trained, and evaluated a model that can achieve high accuracy on the **DigitDataset**. This project demonstrates how a **multi-layer neural network (MLNN)** can be applied to solve a supervised learning task.

## Dataset

The dataset used is MATLAB’s built-in **`DigitDataset`**, which contains images of handwritten digits (0–9). The dataset is split into three subsets:

- **Training and Validation Set:** 80% of the dataset is used for training and validation.
- **Testing Set:** 20% of the dataset is reserved for testing the model's performance.

The dataset is loaded and managed using MATLAB’s **`imageDatastore`** function, which makes it easy to handle large image datasets.

## Model Architecture

The architecture of the CNN consists of several layers that work together to extract features and classify the digits:

1. **Input Layer:** Accepts input images of size 28x28 pixels in grayscale.
2. **Convolutional Layers:** Two convolutional layers with 3x3 filters (16 filters in the first layer and 32 in the second) to extract features from the images.
3. **Batch Normalization & ReLU:** Normalize activations and introduce non-linearity after each convolutional layer.
4. **Max Pooling Layers:** Reduces the spatial dimensions of the images (downsampling) by applying a 2x2 pooling operation.
5. **Fully Connected Layers:** 
   - First fully connected layer with 128 neurons.
   - Second fully connected layer with 10 neurons (one for each digit).
6. **Dropout Layer:** Applies a 50% dropout rate to prevent overfitting.
7. **Softmax Layer:** Converts the raw network output into probabilities for each digit.
8. **Classification Layer:** Assigns the final class label based on the probabilities.

## Training Process

The model was trained with the following hyperparameters:

- **Optimizer:** Adam optimizer
- **Learning Rate:** 0.001
- **Epochs:** 10
- **Mini-Batch Size:** 128
- **Validation:** A separate validation set is used to monitor performance during training.

During training, the model was able to learn from the dataset and improve its performance over time.

### Early Stopping:
The training process includes indirect early stopping by monitoring validation performance at regular intervals.

## Testing and Evaluation

After training, the model was evaluated on the **test set**, and the following metrics were calculated:

- **Accuracy:** The percentage of correctly classified digits.
- **Loss:** The value representing how well the model fits the data.

Additionally, a random test image was displayed along with its **predicted** and **actual labels** for visual inspection.

## Code Overview

The project consists of the following main scripts:

1. **`train_model.m`:** 
   - Loads the dataset, splits it into training, validation, and test sets.
   - Defines the CNN architecture and trains the model using the **`trainNetwork`** function.
   - Saves the trained model for future use.

2. **`test_model.m`:**
   - Loads the trained model from the saved file.
   - Evaluates the model on the test set and calculates its accuracy.
   - Displays a random test image and its predicted label.

3. **`random_predictions.m`:**
   - Makes predictions on random test images.
   - Displays the images along with their predicted labels in a 5x5 grid format.

## Results

### Training:
- **Validation Accuracy:** 99.81% after 10 epochs.
- **Loss:** The training and validation loss steadily decreased, showing effective learning.

### Test Performance:
The model achieved high accuracy on the test dataset, confirming its ability to generalize to new, unseen data.

### Example Test Predictions:
Here are some example predictions on randomly selected test images:
- Predicted label: `7`, Actual label: `7`
- Predicted label: `1`, Actual label: `1`

The accuracy and robustness of the model are demonstrated by the consistent performance across various test images.

## Key Takeaways

- The CNN model performs excellently on the handwritten digit recognition task with an accuracy of **over 99%**.
- MATLAB's deep learning toolbox simplifies the process of building, training, and evaluating complex models.
- The trained model is reusable without retraining, making it efficient for future digit recognition tasks.

