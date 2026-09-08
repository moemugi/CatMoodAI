from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np

import os
import sys
import tempfile
from pathlib import Path

from pydub import AudioSegment

# PATHS

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

MODEL_PATH = SRC_DIR / "cat_mood_transfer_model.keras"
CLASSES_PATH = SRC_DIR / "classes.npy"
MEAN_PATH = SRC_DIR / "normalization_mean.npy"
STD_PATH = SRC_DIR / "normalization_std.npy"


# IMPORT AUDIO PROCESSING
sys.path.append(str(SRC_DIR))

from audio_processing import audio_to_mels

# FASTAPI
app = FastAPI(
    title="Cat Mood AI API",
    description="API for cat vocalization mood classification",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# LOAD MODEL
print("=" * 60)
print("CAT MOOD AI API")
print("=" * 60)

print("\nLoading model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

# LOAD CLASSES
if not CLASSES_PATH.exists():
    raise FileNotFoundError(
        f"Classes file not found: {CLASSES_PATH}"
    )

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print("\nClasses:")

for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")

# LOAD NORMALIZATION VALUES
mean = float(
    np.load(MEAN_PATH)
)

std = float(
    np.load(STD_PATH)
)

print(
    f"\nNormalization mean: {mean:.4f}"
)

print(
    f"Normalization std: {std:.4f}"
)

# SETTINGS
IMG_SIZE = 224

# PREPARE SPECTROGRAMS
def prepare_spectrograms(mels):

    X = np.array(
        mels,
        dtype=np.float32
    )

    # Add channel dimension
    X = X[..., np.newaxis]

    # Normalize using training dataset statistics
    X = (
        X - mean
    ) / (
        std + 1e-8
    )

    # Remove channel temporarily
    X = X[..., 0]

    # Add channel back
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

# HEALTH CHECK
@app.get("/")
def root():

    return {
        "status": "online",
        "message": "Cat Mood AI API is running",
        "model": "MobileNetV2 Transfer Learning"
    }

# MODEL INFORMATION
@app.get("/model-info")
def model_info():

    return {
        "model": "MobileNetV2 Transfer Learning",
        "classes": [
            str(class_name)
            for class_name in classes
        ],
        "number_of_classes": len(classes)
    }

# PREDICT
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # VALIDATE FILE
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file provided."
        )

    # CHECK AUDIO EXTENSION
    allowed_extensions = [
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".flac",
        ".webm"
    ]

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use MP3, WAV, M4A, OGG, or FLAC."
            )
        )

    # SAVE TEMPORARY AUDIO FILE
    temp_path = None
    converted_path = None
    try:

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
        ) as temp_file:

            temp_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)


        # CONVERT WEBM RECORDING TO WAV
        if extension == ".webm":

            print(
                "\nBrowser recording detected."
            )

            print(
                "Converting WebM → WAV..."
            )

            converted_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ).name

            audio = AudioSegment.from_file(
                temp_path,
                format="webm"
            )
            # Convert to format expected
            # by the audio processing pipeline
            audio = (
                audio
                    .set_frame_rate(16000)
                    .set_channels(1)
            )

            audio.export(
                converted_path,
                format="wav"
            )
            processing_path = converted_path

            print(
                "Conversion complete."
            )

        else:
            processing_path = temp_path

        # AUDIO → MEL SPECTROGRAMS
        print("\n" + "=" * 60)

        print(
            "NEW CAT MOOD PREDICTION"
        )

        print("=" * 60)

        print(
            f"File: {file.filename}"
        )

        print(
            "\nConverting audio "
            "to Mel spectrograms..."
        )

        mels = audio_to_mels(
            processing_path
        )

        if len(mels) == 0:

            raise HTTPException(
                status_code=400,
                detail="Could not generate audio segments."
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

        # MODEL PREDICTION
        print(
            "\nGenerating predictions..."
        )

        predictions = model.predict(
            X,
            verbose=0
        )

        # SEGMENT PREDICTIONS
        segment_results = []

        for i, probabilities in enumerate(
            predictions
        ):

            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_class = str(
                classes[predicted_index]
            )

            confidence = float(
                probabilities[predicted_index]
            )

            segment_results.append({
                "segment": i + 1,
                "prediction": predicted_class,
                "confidence": round(
                    confidence * 100,
                    2
                )
            })

        # RECORDING-LEVEL PREDICTION
        # Average probabilities from
        # all audio segments.
        average_probabilities = np.mean(
            predictions,
            axis=0
        )

        final_index = int(
            np.argmax(
                average_probabilities
            )
        )

        final_mood = str(
            classes[final_index]
        )

        final_confidence = float(
            average_probabilities[
                final_index
            ]
        )

        # TOP 3 MOODS
        top_indices = np.argsort(
            average_probabilities
        )[::-1][:3]


        top_3 = []
        for index in top_indices:

            top_3.append({
                "mood": str(
                    classes[index]
                ),

                "confidence": round(
                    float(
                        average_probabilities[index]
                    ) * 100,
                    2
                )
            })

        # RESULT
        result = {

            "filename": file.filename,

            "segments": len(mels),

            "predicted_mood": final_mood,

            "confidence": round(
                final_confidence * 100,
                2
            ),

            "top_3": top_3,

            "segment_predictions":
                segment_results
        }

        # PRINT RESULT
        print("\n" + "=" * 60)

        print(
            "FINAL CAT MOOD"
        )

        print("=" * 60)

        print(
            f"Predicted mood: "
            f"{final_mood}"
        )

        print(
            f"Confidence: "
            f"{final_confidence * 100:.2f}%"
        )

        print(
            "\nPrediction complete."
        )


        return result


    except HTTPException:

        raise

    except Exception as e:
        print(
            f"\nPrediction error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        # Delete original temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        # Delete converted WAV file
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)