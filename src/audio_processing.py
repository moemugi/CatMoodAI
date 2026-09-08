import librosa
import numpy as np


# AUDIO SETTINGS

SAMPLE_RATE = 16000

WINDOW_SECONDS = 4
HOP_SECONDS = 2

N_MELS = 128
HOP_LENGTH = 512


# LOAD AUDIO

def load_audio(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return audio


# SPLIT AUDIO INTO WINDOWS

def split_audio(audio):

    window_length = SAMPLE_RATE * WINDOW_SECONDS
    hop_length = SAMPLE_RATE * HOP_SECONDS

    segments = []

    # SHORT RECORDING

    if len(audio) <= window_length:

        if len(audio) < window_length:

            padding = window_length - len(audio)

            audio = np.pad(
                audio,
                (0, padding),
                mode="constant"
            )

        segments.append(audio)

        return segments

    # LONG RECORDING

    start = 0

    while start + window_length <= len(audio):

        segment = audio[
            start:start + window_length
        ]

        segments.append(segment)

        start += hop_length

    # HANDLE REMAINING AUDIO

    if start < len(audio):

        remaining = audio[start:]

        if len(remaining) > 0:

            padding = window_length - len(remaining)

            remaining = np.pad(
                remaining,
                (0, padding),
                mode="constant"
            )

            segments.append(remaining)

    return segments


# AUDIO → MEL SPECTROGRAM

def segment_to_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# FILE → MEL SPECTROGRAMS

def audio_to_mels(file_path):

    audio = load_audio(file_path)

    segments = split_audio(audio)

    mels = []

    for segment in segments:

        mel = segment_to_mel(segment)

        mels.append(mel)

    return mels