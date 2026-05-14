import random
import numpy as np
import tensorflow as tf
import h5py

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from fine_tuning.DataAugmentation import DataAugmentation


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


HDF5_PATH = "train_data.hdf5"
BASE_MODEL_PATH = "best_model.keras"
FINE_TUNED_MODEL_PATH = "best_finetuned_model.keras"


print("[INFO] Loading data...")

with h5py.File(HDF5_PATH, "r") as f:
    data = f["images"][:]
    labels = f["labels"].asstr()[:]


data = data.astype("float32") / 255.0
data = data[..., np.newaxis]

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

class_names = lb.classes_

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train
)


callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        FINE_TUNED_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]


da = DataAugmentation(
    angle=10,
    width=0.05,
    height=0.05,
    shear=0.05,
    zoom=0.05,
    h_flip=False
)

imageGen = da.imageGenerator(
    X_train,
    y_train,
    batch_size=32
)


model = load_model(BASE_MODEL_PATH)

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)

print("[INFO] Fine-tuning model...")

history = model.fit(
    imageGen,
    validation_data=(X_val, y_val),
    epochs=20,
    callbacks=callbacks
)


pred = model.predict(X_test).argmax(axis=1)
true = y_test.argmax(axis=1)

print(classification_report(true, pred, target_names=class_names))

cm = confusion_matrix(true, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")
plt.title("Fine-tuned Model Confusion Matrix")
plt.show()


model.save("final_finetuned_model.keras")
print("[INFO] Fine-tuned model saved.")
