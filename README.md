# 🧠 Image Classification using CNN (CIFAR-10)

## 📌 Project Purpose

This project demonstrates how to build a Convolutional Neural Network (CNN) for **multi-class image classification** using the popular **CIFAR-10 dataset**. The goal is to train a model that can accurately identify images into one of the 10 predefined object classes.

---

## ❓ Problem Statement

Traditional machine learning models struggle with raw pixel data, especially for image classification tasks due to spatial and scale variations. The challenge is to develop a deep learning model that can learn spatial hierarchies and identify key patterns in images for robust classification.

---

## 🎯 Project Objective

- Build a CNN model from scratch using TensorFlow/Keras.
- Train the model on CIFAR-10 dataset to classify 10 different classes.
- Monitor and improve training and validation accuracy.
- Visualize performance using graphs and prediction samples.
- Achieve test accuracy in the range of **75%–85%**.

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

## 🧱 Model Architecture

- ✅ Input: 32x32x3 RGB images
- ✅ Conv2D + MaxPooling2D layers (multiple blocks)
- ✅ Flatten layer
- ✅ Dense layers with ReLU activation
- ✅ Dropout for regularization
- ✅ Output: Dense layer with 10 units and Softmax activation

---

## 📈 Results

- ✅ Final Validation Accuracy: ~**78%**
- ✅ Final Test Accuracy: ~**76.9%**
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
