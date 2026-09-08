import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2


# SETTINGS
IMG_SIZE = 224

MODEL_PATH = "cat_mood_transfer_model.keras"


# LOAD DATASET
X_test = np.load("X_test.npy")

y_test = np.load("y_test.npy")

test_groups = np.load(
    "test_groups.npy",
    allow_pickle=True
)

classes = np.load(
    "classes.npy",
    allow_pickle=True
)


# PREPARE SPECTROGRAMS
def prepare_spectrograms(X):

    # Remove existing channel dimension
    X = X[..., 0]

    # Add channel dimension
    X = tf.expand_dims(
        X,
        axis=-1
    )
    # Resize for MobileNetV2
    X = tf.image.resize(
        X,
        [IMG_SIZE, IMG_SIZE]
    )
    # Convert grayscale to RGB
    X = tf.image.grayscale_to_rgb(
        X
    )
    return X.numpy()


print("=" * 70)
print("CAT MOOD TRANSFER LEARNING - SEGMENT ANALYSIS")
print("=" * 70)

print(
    f"\nOriginal test recordings: "
    f"{len(np.unique(test_groups))}"
)

print(
    f"Total test segments: "
    f"{len(X_test)}"
)

print(
    "\nPreparing test spectrograms..."
)

X_test_prepared = prepare_spectrograms(
    X_test
)

print(
    f"Prepared testing data: "
    f"{X_test_prepared.shape}"
)


# LOAD MODEL
print(
    "\nLoading transfer-learning model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# GENERATE PREDICTION
print(
    "\nGenerating segment predictions..."
)
probabilities = model.predict(
    X_test_prepared,
    verbose=0
)
predictions = np.argmax(
    probabilities,
    axis=1
)

# SEGMENT-LEVEL ANALYSIS
print("\n")
print("=" * 70)
print("SEGMENT-LEVEL ANALYSIS")
print("=" * 70)


for i in range(len(X_test)):

    actual_class = y_test[i]
    predicted_class = predictions[i]
    confidence = probabilities[
        i,
        predicted_class
    ]
    actual_name = classes[
        actual_class
    ]
    predicted_name = classes[
        predicted_class
    ]
    result = (
        "CORRECT"
        if actual_class == predicted_class
        else "WRONG"
    )
    print("\n" + "-" * 70)
    print(
        f"Segment {i + 1:02d}"
    )
    print(
        f"Recording : "
        f"{test_groups[i]}"
    )
    print(
        f"Actual    : "
        f"{actual_name}"
    )
    print(
        f"Predicted : "
        f"{predicted_name}"
    )
    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )
    print(
        f"Result    : "
        f"{result}"
    )


    # TOP 3 PREDICTIONS
    top_indices = np.argsort(
        probabilities[i]
    )[::-1][:3]

    print("Top 3:")

    for rank, class_index in enumerate(
        top_indices,
        start=1
    ):
        class_name = classes[
            class_index
        ]
        class_probability = probabilities[
            i,
            class_index
        ]
        print(
            f"  {rank}. "
            f"{class_name:<15} "
            f"{class_probability * 100:.2f}%"
        )

# SUMMARY
segment_accuracy = np.mean(
    predictions == y_test
)
print("\n")
print("=" * 70)
print("SEGMENT SUMMARY")
print("=" * 70)

print(
    f"\nTotal segments: "
    f"{len(y_test)}"
)

print(
    f"Correct segments: "
    f"{np.sum(predictions == y_test)}"
)

print(
    f"Wrong segments: "
    f"{np.sum(predictions != y_test)}"
)

print(
    f"Segment-level accuracy: "
    f"{segment_accuracy * 100:.2f}%"
)


# RECORDING SUMMARY
print("\n")
print("=" * 70)
print("RECORDING SEGMENT SUMMARY")
print("=" * 70)

unique_recordings = np.unique(
    test_groups
)

for recording in unique_recordings:
    indices = np.where(
        test_groups == recording
    )[0]
    actual_class = y_test[
        indices[0]
    ]
    recording_predictions = predictions[
        indices
    ]
    print("\n" + "-" * 70)
    print(
        f"Recording: {recording}"
    )
    print(
        f"Actual:    "
        f"{classes[actual_class]}"
    )
    print(
        f"Segments:  "
        f"{len(indices)}"
    )
    for index in indices:
        print(
            f"  Segment {index + 1:02d}: "
            f"{classes[predictions[index]]:<15} "
            f"{probabilities[index, predictions[index]] * 100:.2f}%"
        )

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)