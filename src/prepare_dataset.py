import os
import random
import numpy as np

from audio_processing import audio_to_mels


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "../CAT_DB"

RANDOM_SEED = 42

TRAIN_PER_CLASS = 7
VAL_PER_CLASS = 2
TEST_PER_CLASS = 1


# ==========================================
# RANDOM SEED
# ==========================================

random.seed(RANDOM_SEED)


# ==========================================
# FIND CLASSES
# ==========================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])


class_to_index = {
    class_name: index
    for index, class_name in enumerate(classes)
}


print("=" * 60)
print("CAT MOOD DATASET PREPARATION")
print("=" * 60)

print(f"\nClasses: {len(classes)}")

for index, class_name in enumerate(classes):
    print(f"{index}: {class_name}")


# ==========================================
# DATA STORAGE
# ==========================================

X_train = []
y_train = []
train_groups = []

X_val = []
y_val = []
val_groups = []

X_test = []
y_test = []
test_groups = []


# ==========================================
# PROCESS AUDIO FILES
# ==========================================

def process_files(
    files,
    folder_path,
    label,
    X,
    y,
    groups
):

    for file in files:

        file_path = os.path.join(
            folder_path,
            file
        )

        try:

            # Convert recording into
            # one or more Mel spectrograms
            mels = audio_to_mels(
                file_path
            )

            for mel in mels:

                X.append(mel)

                y.append(label)

                # IMPORTANT:
                # Remember which original
                # recording produced this segment
                groups.append(file)

            print(
                f"  {file} → "
                f"{len(mels)} segment(s)"
            )

        except Exception as e:

            print(
                f"  ERROR: {file}"
            )

            print(
                f"  {e}"
            )


# ==========================================
# PROCESS EACH CLASS
# ==========================================

for class_name in classes:

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    audio_files = [
        file
        for file in os.listdir(folder_path)
        if file.lower().endswith(".mp3")
    ]

    random.shuffle(audio_files)

    label = class_to_index[class_name]

    # ======================================
    # SPLIT ORIGINAL RECORDINGS
    # ======================================

    train_files = audio_files[
        :TRAIN_PER_CLASS
    ]

    val_files = audio_files[
        TRAIN_PER_CLASS:
        TRAIN_PER_CLASS + VAL_PER_CLASS
    ]

    test_files = audio_files[
        TRAIN_PER_CLASS + VAL_PER_CLASS:
    ]


    print("\n" + "-" * 60)
    print(class_name)
    print("-" * 60)

    print(
        f"Original recordings: "
        f"{len(audio_files)}"
    )


    # ======================================
    # TRAINING
    # ======================================

    print("\nTraining:")

    process_files(
        train_files,
        folder_path,
        label,
        X_train,
        y_train,
        train_groups
    )


    # ======================================
    # VALIDATION
    # ======================================

    print("\nValidation:")

    process_files(
        val_files,
        folder_path,
        label,
        X_val,
        y_val,
        val_groups
    )


    # ======================================
    # TESTING
    # ======================================

    print("\nTesting:")

    process_files(
        test_files,
        folder_path,
        label,
        X_test,
        y_test,
        test_groups
    )


# ==========================================
# CONVERT TO NUMPY
# ==========================================

X_train = np.array(
    X_train,
    dtype=np.float32
)

y_train = np.array(
    y_train
)

train_groups = np.array(
    train_groups
)


X_val = np.array(
    X_val,
    dtype=np.float32
)

y_val = np.array(
    y_val
)

val_groups = np.array(
    val_groups
)


X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test
)

test_groups = np.array(
    test_groups
)


# ==========================================
# ADD CHANNEL DIMENSION
# ==========================================

X_train = X_train[
    ..., np.newaxis
]

X_val = X_val[
    ..., np.newaxis
]

X_test = X_test[
    ..., np.newaxis
]


# ==========================================
# NORMALIZATION
# ==========================================

mean = X_train.mean()

std = X_train.std()


X_train = (
    X_train - mean
) / (std + 1e-8)


X_val = (
    X_val - mean
) / (std + 1e-8)


X_test = (
    X_test - mean
) / (std + 1e-8)


# ==========================================
# FINAL DATASET INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(
    f"\nTraining:"
    f"    X = {X_train.shape}"
    f"    y = {y_train.shape}"
)

print(
    f"Validation:"
    f"  X = {X_val.shape}"
    f"    y = {y_val.shape}"
)

print(
    f"Testing:"
    f"     X = {X_test.shape}"
    f"    y = {y_test.shape}"
)


# ==========================================
# GROUP INFORMATION
# ==========================================

print("\n" + "-" * 60)
print("SEGMENT GROUP INFORMATION")
print("-" * 60)

print(
    f"Training recordings: "
    f"{len(np.unique(train_groups))}"
)

print(
    f"Validation recordings: "
    f"{len(np.unique(val_groups))}"
)

print(
    f"Testing recordings: "
    f"{len(np.unique(test_groups))}"
)


# ==========================================
# NORMALIZATION INFORMATION
# ==========================================

print(
    f"\nNormalization mean: "
    f"{mean:.4f}"
)

print(
    f"Normalization std:  "
    f"{std:.4f}"
)


# ==========================================
# SAVE DATASET
# ==========================================

np.save(
    "X_train.npy",
    X_train
)

np.save(
    "y_train.npy",
    y_train
)

np.save(
    "train_groups.npy",
    train_groups
)


np.save(
    "X_val.npy",
    X_val
)

np.save(
    "y_val.npy",
    y_val
)

np.save(
    "val_groups.npy",
    val_groups
)


np.save(
    "X_test.npy",
    X_test
)

np.save(
    "y_test.npy",
    y_test
)

np.save(
    "test_groups.npy",
    test_groups
)


np.save(
    "classes.npy",
    np.array(classes)
)

np.save(
    "normalization_mean.npy",
    np.array(mean)
)

np.save(
    "normalization_std.npy",
    np.array(std)
)


# ==========================================
# VERIFY GROUPS
# ==========================================

print("\n" + "-" * 60)
print("TEST RECORDING GROUPS")
print("-" * 60)

for group in np.unique(test_groups):

    count = np.sum(
        test_groups == group
    )

    print(
        f"{group}: "
        f"{count} segment(s)"
    )


print("\nDataset files saved successfully!")

print("=" * 60)