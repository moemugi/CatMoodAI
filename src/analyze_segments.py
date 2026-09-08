import numpy as np
import tensorflow as tf

# LOAD DATA
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

# LOAD MODEL
model = tf.keras.models.load_model(
    "cat_mood_transfer_model.keras"
)

# PREPARE DATA FOR MOBILENETV2
print("Preparing test spectrograms...")

X_test_rgb = np.repeat(
    X_test,
    3,
    axis=-1
)

X_test_rgb = tf.image.resize(
    X_test_rgb,
    (224, 224)
)

X_test_rgb = X_test_rgb.numpy()

# PREDICTIONS
print("Generating segment predictions...")

probabilities = model.predict(
    X_test_rgb,
    verbose=0
)
predictions = np.argmax(
    probabilities,
    axis=1
)

# DISPLAY SEGMENTS
print("\n" + "=" * 70)
print("SEGMENT-LEVEL ANALYSIS")
print("=" * 70)


for i in range(len(X_test)):
    actual = classes[y_test[i]]
    predicted = classes[predictions[i]]
    confidence = probabilities[i][predictions[i]]
    result = (
        "CORRECT"
        if y_test[i] == predictions[i]
        else "WRONG"
    )
    print(
        f"\nSegment {i + 1:02d}"
    )
    print(
        f"Recording : {test_groups[i]}"
    )
    print(
        f"Actual    : {actual}"
    )
    print(
        f"Predicted : {predicted}"
    )
    print(
        f"Confidence: {confidence * 100:.2f}%"
    )
    print(
        f"Result    : {result}"
    )

    # TOP 3 PREDICTIONS
    top_indices = np.argsort(
        probabilities[i]
    )[::-1][:3]

    print("Top 3:")
    for rank, index in enumerate(
        top_indices,
        start=1
    ):
        print(
            f"  {rank}. "
            f"{classes[index]:<15} "
            f"{probabilities[i][index] * 100:.2f}%"
        )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)