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

_________________________________________________________________

Layer (type)                 Output Shape              Param #

=================================================================

conv2d (Conv2D)              (None, 30, 30, 32)        896

max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0

flatten (Flatten)            (None, 7200)              0

dense (Dense)                (None, 128)               921728

dropout (Dropout)            (None, 128)               0

dense_1 (Dense)              (None, 10)                1290

=================================================================

Total params: 923,914

Trainable params: 923,914

Non-trainable params: 0

---

## 📈 Results

- ✅ Final Validation Accuracy: ~**79%**
- ✅ Final Test Accuracy: ~**78.8%**
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

git clone https://github.com/shivharebhupendra/image-classification-cnn.git
cd image-classification-cnn

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
