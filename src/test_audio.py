from audio_processing import (
    load_audio,
    split_audio,
    segment_to_mel
)


# ==========================================
# AUDIO FILE
# ==========================================

file_path = "../CAT_DB/Happy/cat_68.mp3"


# ==========================================
# LOAD
# ==========================================

audio = load_audio(file_path)


# ==========================================
# SPLIT
# ==========================================

segments = split_audio(audio)


# ==========================================
# DISPLAY
# ==========================================

print("=" * 60)
print("AUDIO SEGMENTATION TEST")
print("=" * 60)

print(
    f"Original duration: "
    f"{len(audio) / 16000:.2f} seconds"
)

print(
    f"Number of segments: "
    f"{len(segments)}"
)


# ==========================================
# TEST MEL
# ==========================================

for i, segment in enumerate(segments):

    mel = segment_to_mel(segment)

    print(
        f"Segment {i + 1}: "
        f"{len(segment) / 16000:.2f}s "
        f"→ {mel.shape}"
    )


print("=" * 60)