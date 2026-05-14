import random
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models.models import VGGNet


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


HDF5_PATH = "train_data.hdf5"
MODEL_PATH = "best_model.keras"


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

print("[INFO] Train shape:", X_train.shape)
print("[INFO] Val shape:", X_val.shape)
print("[INFO] Test shape:", X_test.shape)
print("[INFO] Classes:", class_names)


model = VGGNet.build(
    height=X_train.shape[1],
    width=X_train.shape[2],
    depth=1,
    classes=len(class_names)
)

opt = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]

print("[INFO] Training model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=100,
    callbacks=callbacks
)

print("[INFO] Evaluating model...")

pred = model.predict(X_test).argmax(axis=1)
true = y_test.argmax(axis=1)

print(classification_report(true, pred, target_names=class_names))

cm = confusion_matrix(true, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


model.save("final_model.keras")
print("[INFO] Model saved as final_model.keras")
