import h5py
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


MODEL_PATH = "best_model.keras"          # or final_model.keras
TEST_HDF5_PATH = "test_data.hdf5"


with h5py.File(TEST_HDF5_PATH, "r") as f:
    X_test = f["images"][:]
    y_test_labels = f["labels"].asstr()[:]


X_test = X_test.astype("float32") / 255.0
X_test = X_test[..., np.newaxis]


le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test_labels)
y_test = to_categorical(y_test_encoded)

class_names = le.classes_

model = load_model(MODEL_PATH)

pred_probs = model.predict(X_test)
pred = pred_probs.argmax(axis=1)
true = y_test.argmax(axis=1)

print("Classes:", class_names)
print(classification_report(true, pred, target_names=class_names))

cm = confusion_matrix(true, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()
plt.title("Test Confusion Matrix")
plt.show()
