import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

# SETTINGS
MODEL_PATH = "cat_mood_transfer_model.keras"

# LOAD DATASET
X_test = np.load(
    "X_test.npy"
)

y_test = np.load(
    "y_test.npy"
)

test_groups = np.load(
    "test_groups.npy",
    allow_pickle=True
)

classes = np.load(
    "classes.npy",
    allow_pickle=True
)

# DISPLAY DATASET
print("=" * 60)
print("CAT MOOD TRANSFER LEARNING EVALUATION")
print("=" * 60)

print(
    f"\nOriginal test recordings: "
    f"{len(np.unique(test_groups))}"
)

print(
    f"Total test segments: "
    f"{len(X_test)}"
)

# PREPARE SPECTROGRAMS FOR MOBILENETV2
print(
    "\nPreparing test spectrograms "
    "for MobileNetV2..."
)

# Remove channel dimension
X_test_2d = X_test[..., 0]

# Convert to TensorFlow tensor
X_test_tensor = tf.convert_to_tensor(
    X_test_2d,
    dtype=tf.float32
)

# Add channel dimension
X_test_tensor = tf.expand_dims(
    X_test_tensor,
    axis=-1
)

# Resize from:
# (128, 126, 1)
#
# to:
# (224, 224, 1)

X_test_tensor = tf.image.resize(
    X_test_tensor,
    [224, 224]
)

# CONVERT GRAYSCALE TO RGB
X_test_tensor = tf.image.grayscale_to_rgb(
    X_test_tensor
)

X_test = X_test_tensor.numpy()

print(
    f"Prepared testing data: "
    f"{X_test.shape}"
)

# LOAD TRAINED TRANSFER MODEL
model = tf.keras.models.load_model(
    MODEL_PATH
)

# SEGMENT PREDICTIONS
print(
    "\nGenerating predictions..."
)

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

# GROUP SEGMENTS BY ORIGINAL RECORDING
recordings = {}
for i in range(len(test_groups)):

    filename = test_groups[i]

    if filename not in recordings:

        recordings[filename] = {
            "actual": y_test[i],
            "probabilities": []
        }

    recordings[
        filename
    ][
        "probabilities"
    ].append(
        probabilities[i]
    )

# COMBINE SEGMENTS
recording_actual = []
recording_predictions = []
recording_confidences = []
recording_names = []

for filename, data in recordings.items():

    segment_probabilities = np.array(
        data["probabilities"]
    )
    # Average probabilities
    average_probability = np.mean(
        segment_probabilities,
        axis=0
    )
    # Predicted class
    predicted_class = np.argmax(
        average_probability
    )
    # Confidence
    confidence = average_probability[
        predicted_class
    ]
    # Actual class
    actual_class = data["actual"]

    recording_names.append(
        filename
    )

    recording_actual.append(
        actual_class
    )

    recording_predictions.append(
        predicted_class
    )

    recording_confidences.append(
        confidence
    )

# CONVERT TO NUMPY
recording_actual = np.array(
    recording_actual
)

recording_predictions = np.array(
    recording_predictions
)

recording_confidences = np.array(
    recording_confidences
)

# RECORDING-LEVEL ACCURACY
accuracy = np.mean(
    recording_predictions
    ==
    recording_actual
)

# DISPLAY RESULTS
print("\n" + "=" * 60)

print(
    "CAT MOOD TRANSFER LEARNING EVALUATION"
)

print("=" * 60)

print(
    f"\nOriginal test recordings: "
    f"{len(recordings)}"
)

print(
    f"Total test segments: "
    f"{len(X_test)}"
)

print(
    f"\nRecording-Level Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# RECORDING PREDICTIONS
print(
    "\nRECORDING-LEVEL PREDICTIONS"
)

print("-" * 60)

for i in range(
    len(recording_names)
):

    actual = classes[
        recording_actual[i]
    ]
    predicted = classes[
        recording_predictions[i]
    ]
    confidence = (
        recording_confidences[i]
    )
    result = (
        "CORRECT"
        if recording_actual[i]
        ==
        recording_predictions[i]
        else
        "WRONG"
    )
    print(
        f"{i + 1:02d}. "
        f"{recording_names[i]:<25} "
        f"Actual: {actual:<15} "
        f"Predicted: {predicted:<15} "
        f"Confidence: "
        f"{confidence * 100:5.1f}% "
        f"{result}"
    )

# CONFUSION MATRIX
cm = confusion_matrix(
    recording_actual,
    recording_predictions,
    labels=range(
        len(classes)
    )
)

print(
    "\nCONFUSION MATRIX"
)

print("-" * 60)

print(cm)

# CLASSIFICATION REPORT
print(
    "\nCLASSIFICATION REPORT"
)

print("-" * 60)

print(
    classification_report(
        recording_actual,
        recording_predictions,
        labels=range(
            len(classes)
        ),
        target_names=classes,
        zero_division=0
    )
)

# DISPLAY CONFUSION MATRIX
plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    cm
)

plt.title(
    "Cat Mood Transfer Learning "
    "Confusion Matrix - Recording Level"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(classes)),
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(classes)),
    classes
)

# ADD NUMBERS TO CELLS
for i in range(
    len(classes)
):

    for j in range(
        len(classes)
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()
plt.tight_layout()
plt.show()