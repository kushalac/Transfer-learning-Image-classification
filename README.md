# Transfer Learning for Image Classification

This project implements transfer learning using pre-trained models like ResNet50, ResNet101, EfficientNetB0, and VGG16 to classify images of six scenes. The dataset is preprocessed and augmented using various techniques to enhance model performance. The final models are evaluated on training, validation, and test data for metrics like Precision, Recall, AUC, and F1 score.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Model Architecture](#model-architecture)
4. [Image Augmentation](#image-augmentation)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [References](#references)

---

## Project Overview

In this project:
- Pre-trained models are used as feature extractors.
- Only the last few layers are fine-tuned for classification.
- Empirical regularization techniques (like cropping, zooming, rotation) are applied to enhance robustness.
- Metrics like Precision, Recall, AUC, and F1 score are reported to assess the models.

---

## Data

The dataset contains images of six scenes organized into separate folders for training and testing.  
- Images are resized to 224x224 pixels.
- A random subset of 20% of the training data is used for validation.

Data Preprocessing includes:
- One-hot encoding for class labels.
- Zero-padding or resizing of images to ensure uniformity.

---

## Model Architecture

We use the following pre-trained models:
1. **ResNet50**
2. **ResNet101**
3. **EfficientNetB0**
4. **VGG16**

The architecture includes:
- Frozen layers from the pre-trained models.
- A global average pooling layer.
- Batch normalization.
- Dropout with a rate of 20%.
- Fully connected layers with ReLU activation and softmax for output.

The optimizer used is **Adam**, and the loss function is **categorical cross-entropy**.

---

## Image Augmentation

To improve generalization, the following augmentations are applied:
- Random rotations (up to 30 degrees)
- Width and height shifts (up to 20%)
- Random zooming (up to 20%)
- Horizontal flipping
- Brightness adjustments

These augmentations are implemented using the `ImageDataGenerator` from Keras.

---

## Training and Evaluation

### Training
- The models are trained for at least **50 epochs**, with **early stopping** to avoid overfitting.
- The batch size used is **16**.

### Metrics
We evaluate the models using:
- **Precision**
- **Recall**
- **AUC**
- **F1 Score**

Evaluation is done on training, validation, and test datasets.

---

## Results

| Model         | Training AUC | Validation AUC | Test AUC | Precision | Recall | F1 Score |
|---------------|--------------|----------------|----------|-----------|--------|----------|
| ResNet50      | xx.xx        | xx.xx          | xx.xx    | xx.xx     | xx.xx  | xx.xx    |
| ResNet101     | xx.xx        | xx.xx          | xx.xx    | xx.xx     | xx.xx  | xx.xx    |
| EfficientNetB0| xx.xx        | xx.xx          | xx.xx    | xx.xx     | xx.xx  | xx.xx    |
| VGG16         | xx.xx        | xx.xx          | xx.xx    | xx.xx     | xx.xx  | xx.xx    |

(Replace `xx.xx` with actual results after training.)

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
