# Tumor Detection

Tumor Detection is a deep learning project for classifying brain tumor images using a custom preprocessing, HDF5 dataset storage, model training, and fine-tuning pipeline.

The project uses TensorFlow/Keras for model training, OpenCV for image preprocessing, h5py for storing the processed dataset, and scikit-learn for evaluation.

## Project structure

```text
TumorDetection/
├── best_model.keras
├── save.hdf5
├── requirments.txt
├── ToHDF5.py
├── trainningModel.py
├── tuning.py
├── dataset/
│   └── DatasetLoader.py
├── fine_tuning/
│   └── DataAugmentation.py
├── models/
│   └── models.py
└── preprocessing/
    ├── AspectAwarePreprocess.py
    └── ToGray.py
```

## Main files

### `ToHDF5.py`
Converts the image dataset into an HDF5 file. This is useful when the dataset is large, because HDF5 allows images and labels to be stored efficiently in one file instead of loading thousands of separate images manually every time.

### `dataset/DatasetLoader.py`
Loads images and labels from the HDF5 file. It is responsible for reading the saved dataset and preparing it for training and testing.

### `preprocessing/AspectAwarePreprocess.py`
Resizes images while preserving their aspect ratio. This helps avoid image distortion before feeding data into the neural network.

### `preprocessing/ToGray.py`
Converts images to grayscale. This reduces input complexity and can be useful when color is not the main signal for classification.

### `models/models.py`
Contains the neural network architecture used for classification.

### `trainningModel.py`
Main training script. It loads the dataset, trains the model, validates it, and saves the best model.

### `tuning.py`
Fine-tuning script. It loads the previously trained model and continues training, usually with a smaller learning rate and/or augmented data.

### `fine_tuning/DataAugmentation.py`
Contains data augmentation logic. Augmentation helps the model generalize better by creating modified versions of training images.

### `best_model.keras`
Saved Keras model with the best validation performance.

### `save.hdf5`
Processed dataset stored in HDF5 format.

## Requirements

Recommended Python version:

```text
Python 3.11 or Python 3.12
```

Python 3.14 is not recommended because TensorFlow may not have compatible wheels for it.

Install dependencies:

```bash
pip install -r requirements.txt
```

If the project contains `requirments.txt` instead of `requirements.txt`, either rename it or run:

```bash
pip install -r requirments.txt
```

Recommended `requirements.txt`:

```txt
numpy<2
h5py>=3.10
imutils==0.5.4
scikit-learn>=1.4
opencv-python>=4.9
tensorflow>=2.16
keras>=3.0
matplotlib>=3.8
```

## Setup

Create and activate a virtual environment:

```bash
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, use Python 3.11:

```bash
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How to run

### 1. Convert dataset to HDF5

Run:

```bash
python ToHDF5.py
```

This should create or update the HDF5 dataset file, for example:

```text
save.hdf5
```

### 2. Train the model

Run:

```bash
python trainningModel.py
```

The script should train the model and save the best version as:

```text
best_model.keras
```

### 3. Fine-tune the model

Run:

```bash
python tuning.py
```

Fine-tuning loads the saved model and continues training, often using data augmentation.

## Evaluation

The model can be evaluated using `classification_report` from scikit-learn:

```python
from sklearn.metrics import classification_report

pred = model.predict(X_test).argmax(axis=1)
true = y_test.argmax(axis=1)

print(classification_report(true, pred))
```

Example metrics from testing:

```text
accuracy:      ~0.90 - 0.95
macro avg f1:  ~0.92 - 0.95
weighted f1:   ~0.90 - 0.95
```

These results indicate that the model performs well overall, but class-level performance should be checked with a confusion matrix, especially if one class has lower recall.

## Recommended checks

Use a confusion matrix to see which classes are being confused:

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(true, pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

If class `0` is often predicted as class `1`, consider:

- reducing augmentation strength;
- using a smaller fine-tuning learning rate, such as `1e-5`;
- checking class balance;
- reviewing train/validation/test split;
- checking for duplicate or near-duplicate images.

## Fine-tuning tips

For fine-tuning, avoid a large learning rate. A good starting point is:

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=1e-5)
```

Also avoid overwriting the original best model:

```python
ModelCheckpoint(
    'best_finetuned_model.keras',
    monitor='val_loss',
    save_best_only=True
)
```

This keeps the original trained model safe while saving the fine-tuned version separately.

## Notes

- `save.hdf5` stores the processed dataset.
- `best_model.keras` stores the trained model.
- The filename `trainningModel.py` appears to contain a typo; a cleaner name would be `training_model.py`.
- The filename `requirments.txt` also appears to contain a typo; the standard name is `requirements.txt`.

## Possible improvements

Future improvements may include:

- adding command-line arguments for dataset path, image size, batch size, and epochs;
- adding a separate `predict.py` script for inference on new images;
- saving label names together with the model;
- adding confusion matrix visualization;
- adding training history plots;
- separating configuration into a `config.py` file;
- adding a Jupyter notebook demo;
- improving README with dataset source and example images.

## Author

This project was developed as a brain tumor detection/classification pipeline using Python, TensorFlow/Keras, OpenCV, h5py, and scikit-learn.
