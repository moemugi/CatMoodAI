import os
import sys
import numpy as np
import tensorflow as tf

from audio_processing import audio_to_mels


# SETTINGS
MODEL_PATH = "cat_mood_transfer_model.keras"

IMG_SIZE = 224


# LOAD CLASSES
classes = np.load(
    "classes.npy",
    allow_pickle=True
)


# LOAD NORMALIZATION VALUES
normalization_mean = float(
    np.load("normalization_mean.npy")
)

normalization_std = float(
    np.load("normalization_std.npy")
)


# PREPARE SPECTROGRAMS
def prepare_spectrograms(mels):
    X = np.array(
        mels,
        dtype=np.float32
    )

    # Add channel dimension
    X = X[..., np.newaxis]

    # Normalize using training statistics
    X = (
        X - normalization_mean
    ) / (
        normalization_std + 1e-8
    )

    # Remove channel temporarily
    X = X[..., 0]

    # Add channel again
    X = tf.expand_dims(
        X,
        axis=-1
    )

    # Resize to MobileNetV2 input
    X = tf.image.resize(
        X,
        [IMG_SIZE, IMG_SIZE]
    )

    # Convert grayscale → RGB
    X = tf.image.grayscale_to_rgb(
        X
    )

    return X.numpy()


# PREDICT AUDIO
def predict_audio(file_path):
    print("\n" + "=" * 60)
    print("CAT MOOD PREDICTION")
    print("=" * 60)
    print(
        f"\nAudio file: {file_path}"
    )

    # CHECK FILE
    if not os.path.exists(file_path):

        print(
            "\nERROR: Audio file not found."
        )

        return

    # LOAD AUDIO
    print(
        "\nConverting audio to "
        "Mel spectrograms..."
    )

    mels = audio_to_mels(
        file_path
    )

    print(
        f"Generated segments: "
        f"{len(mels)}"
    )

    # PREPARE INPUT
    X = prepare_spectrograms(
        mels
    )

    print(
        f"Prepared input: "
        f"{X.shape}"
    )

    # LOAD MODEL
    print(
        "\nLoading model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # PREDICT
    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        X,
        verbose=0
    )

    # SEGMENT PREDICTIONS
    print("\n")
    print("=" * 60)
    print("SEGMENT PREDICTIONS")
    print("=" * 60)

    for i, probabilities in enumerate(
        predictions
    ):

        predicted_index = np.argmax(
            probabilities
        )

        confidence = (
            probabilities[predicted_index]
            * 100
        )

        print(
            f"\nSegment {i + 1}"
        )

        print(
            f"Prediction: "
            f"{classes[predicted_index]}"
        )

        print(
            f"Confidence: "
            f"{confidence:.2f}%"
        )

    # RECORDING-LEVEL PREDICTION

    # Average probabilities from
    # all segments.
    average_probabilities = np.mean(
        predictions,
        axis=0
    )

    predicted_index = np.argmax(
        average_probabilities
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = (
        average_probabilities[
            predicted_index
        ] * 100
    )

    # TOP 3 PREDICTIONS
    top_indices = np.argsort(
        average_probabilities
    )[::-1][:3]

    print("\n")
    print("=" * 60)
    print("FINAL CAT MOOD")
    print("=" * 60)

    print(
        f"\nPredicted mood: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    print("\nTop 3 moods:")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        probability = (
            average_probabilities[index]
            * 100
        )

        print(
            f"{rank}. "
            f"{classes[index]:<15}"
            f"{probability:.2f}%"
        )

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)


# COMMAND LINE
if __name__ == "__main__":
    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python predict.py "
            "path_to_audio.mp3"
        )

        print(
            "\nExample:"
        )

        print(
            "python predict.py "
            "../CAT_DB/Happy/cat_68.mp3"
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    predict_audio(
        audio_file
    )