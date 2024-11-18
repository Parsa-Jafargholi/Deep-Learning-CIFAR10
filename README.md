# Deep Learning with CIFAR-10 Dataset

This project demonstrates deep learning techniques for image classification using the CIFAR-10 dataset. It implements two popular architectures, **GoogLeNet** and **ResNet**, to classify images into one of 10 categories.

---

## Dataset Overview

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is widely used for training machine learning and computer vision models.

- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

**Classes:**
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Objectives

The main goals of this project are:

1. **Dataset Preparation:**
   - Download and preprocess the CIFAR-10 dataset.
   - Normalize images for consistent training.
2. **Model Implementation:**
   - Build and train **GoogLeNet** using inception modules.
   - Implement **ResNet-18** using residual connections for enhanced performance.
3. **Model Training and Evaluation:**
   - Train both models on the CIFAR-10 dataset.
   - Evaluate their performance and calculate accuracy on test data.
4. **Visualization:**
   - Display sample images from the dataset.
   - Plot training loss and accuracy metrics for analysis.

---

## Tools and Libraries

- Python
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Project Steps

1. **Data Loading:**
   - Download the CIFAR-10 dataset using TorchVision.
   - Normalize images for consistent input to the models.
   - Create data loaders for training and testing.

2. **Model Implementation:**
   - **GoogLeNet:** Build a deep convolutional neural network using inception modules.
   - **ResNet-18:** Implement a residual learning framework with skip connections.

3. **Training:**
   - Use the **CrossEntropyLoss** function and **Adam** optimizer for training.
   - Train both models for 2 epochs (configurable).

4. **Evaluation:**
   - Calculate accuracy on test data.
   - Display sample predictions to evaluate model performance visually.

---

## Results

- **GoogLeNet:**
  - Achieved an accuracy of **X%** on the test set.
  
- **ResNet-18:**
  - Achieved an accuracy of **Y%** on the test set.

Replace **X** and **Y** with the actual results from your training.

---

## Conclusion

This project highlights the application of two popular deep learning architectures—GoogLeNet and ResNet—on the CIFAR-10 dataset. It demonstrates the strengths of inception modules and residual connections in improving model performance.

---

For a detailed walkthrough of the code and analysis, refer to the [Jupyter Notebook](https://github.com/Parsa-Jafargholi/Deep-Learning-CIFAR10/blob/main/DL.ipynb).
