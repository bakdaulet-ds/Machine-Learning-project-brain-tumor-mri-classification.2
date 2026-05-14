# 🧠 Brain Tumor Detection

A deep learning project for classifying brain tumor MRI images using a complete image preprocessing, HDF5 dataset storage, model training, and fine-tuning pipeline.

This project uses **TensorFlow/Keras** for model training, **OpenCV** for image preprocessing, **h5py** for efficient dataset storage, and **scikit-learn** for model evaluation.

---

## 📌 Project Overview

Brain tumor detection is an important medical imaging task where machine learning can help support faster and more accurate diagnosis.

In this project, MRI images are preprocessed, converted into grayscale format, stored in an HDF5 file, and used to train a neural network model for tumor image classification.

The workflow includes:

- Image preprocessing
- Aspect-aware resizing
- Grayscale conversion
- Dataset storage using HDF5
- Model training
- Fine-tuning
- Model evaluation

---
## Model Performance

### Base Model Results

![Model Performance](images/model_performance.png)

The base CNN model achieved approximately 94% accuracy on the test dataset.  
The confusion matrix shows strong classification performance across all four tumor classes.

---

### Fine-Tuned Model Results

![Fine Tuned Model](images/fine_tuned_model_performance.png)

The fine-tuned model achieved approximately 92% accuracy.  
Although fine-tuning improved some class predictions, the overall accuracy was slightly lower compared to the base model.

--- 

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- h5py
- scikit-learn
- Matplotlib

---

