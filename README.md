# 🧠 Image Classification using CNN (CIFAR-10)

## 📌 Project Purpose

The purpose of this project is to build a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes.

---

## ❓ Problem Statement

Traditional image classification approaches often struggle with low accuracy and high computational costs when working with large image datasets. This project aims to develop a deep learning model that can effectively learn spatial hierarchies and features in images to accurately predict the correct object class in unseen images.

---

## 🎯 Objectives

- Load and preprocess CIFAR-10 image data.
- Build a CNN model using TensorFlow/Keras.
- Train the model and evaluate its performance using validation and test sets.
- Use techniques like Dropout and Pooling to prevent overfitting and enhance accuracy.
- Visualize training/validation performance and predictions.

---

## 📊 Dataset: CIFAR-10

- 60,000 color images (32x32 pixels)
- 10 classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

- **Training set:** 50,000 images  
- **Test set:** 10,000 images

Dataset is available via `keras.datasets.cifar10`.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- CNN (Conv2D, MaxPooling, Flatten, Dense)
- Google Colab / Jupyter Notebook

---

## 🧱 Model Architecture

- ✅ Input: 32x32x3 RGB images
- ✅ Conv2D + MaxPooling2D layers (multiple blocks)
- ✅ Flatten layer
- ✅ Dense layers with ReLU activation
- ✅ Dropout for regularization
- ✅ Output: Dense layer with 10 units and Softmax activation

---

## 📊 Model Summary

Model: "sequential"

| Layer (type)             | Output Shape       | Param #   |
|--------------------------|--------------------|-----------|
| Conv2D                   | (32, 32, 32)        | 896       |
| Conv2D                   | (32, 32, 32)        | 9,248     |
| MaxPooling2D             | (16, 16, 32)        | 0         |
| Dropout                  | (16, 16, 32)        | 0         |
| Conv2D                   | (16, 16, 64)        | 18,496    |
| Conv2D                   | (16, 16, 64)        | 36,928    |
| MaxPooling2D             | (8, 8, 64)          | 0         |
| Dropout                  | (8, 8, 64)          | 0         |
| Flatten                  | (4096)              | 0         |
| Dense                    | (512)               | 2,097,664 |
| Dropout                  | (512)               | 0         |
| Dense                    | (10)                | 5,130     |
| **Total Parameters**     |                    | **2,168,362** |
| Trainable Parameters     |                    | 2,168,362 |
| Non-trainable Parameters |                    | 0         |


---

## 📈 Results

- ✅ Final Validation Accuracy: ~**80%**
- ✅ Final Test Accuracy: ~**79%**
- ✅ Model performs well across most CIFAR-10 classes

---

## 📸 Prediction Sample

- python: 
 - predict_image(index=5)  # shows prediction and actual label with image

---

## 🖼️ Output:

- Predicted: cat | Actual: cat

---

## 🛠️ How to Run

1. Clone the repository:

git clone https://github.com/shivharebhupendra/Image-Classification-with-CNN.git
cd Image-Classification-with-CNN

2. Install dependencies:

pip install -r requirements.txt

3. Run the training notebook:
- Open the Jupyter notebook or .py file.
- Execute cells to load data, build model, train and evaluate.

4. Predict on test images:

predict_image(12)  # function to visualize prediction

---

## ⚙️ Requirements
- Python
- TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook

Install them via:
- pip install tensorflow numpy matplotlib

## 👨‍💻 Author

Bhupendra Shivhare

Aspiring Data Scientist | Deep Learning Enthusiast

LinkedIn: www.linkedin.com/in/bhupendra-shivhare-a8a02a25b

📧 Email: shivharebhupendra@gmail.com
