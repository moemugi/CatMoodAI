import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# AUDIO FILE
file_path = "../CAT_DB/Happy/cat_68.mp3"


# LOAD AUDIO
audio, sr = librosa.load(
    file_path,
    sr=16000,
    mono=True
)


# CREATE MEL SPECTROGRAM
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128,
    hop_length=512
)


mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)


# DISPLAY

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(
    format="%+2.0f dB"
)

plt.title(
    "Mel Spectrogram - Happy Cat"
)

plt.tight_layout()

plt.show()