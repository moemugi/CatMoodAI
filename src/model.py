import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks

# SETTINGS
EPOCHS = 50
BATCH_SIZE = 8

# LOAD DATASET

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

classes = np.load(
    "classes.npy",
    allow_pickle=True
)


# DISPLAY DATASET
print("=" * 60)
print("CAT MOOD CNN TRAINING")
print("=" * 60)

print(f"Training data:   {X_train.shape}")
print(f"Validation data: {X_val.shape}")
print(f"Testing data:    {X_test.shape}")

print(f"Classes: {len(classes)}")

print("\nClass names:")

for i, class_name in enumerate(classes):
    print(f"{i}: {class_name}")


# DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomZoom(
        height_factor=0.05,
        width_factor=0.05
    )

])


# BUILD CNN
model = models.Sequential([

    layers.Input(
        shape=X_train.shape[1:]
    ),

    # --------------------------------------
    # AUGMENTATION
    # --------------------------------------

    data_augmentation,

    # --------------------------------------
    # CONVOLUTION BLOCK 1
    # --------------------------------------

    layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CONVOLUTION BLOCK 2
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # CONVOLUTION BLOCK 3
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # FEATURE REDUCTION
    layers.GlobalAveragePooling2D(),

    # CLASSIFIER
    layers.Dense(
        32,
        activation="relu"
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        len(classes),
        activation="softmax"
    )

])


# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)

# CALLBACKS
early_stopping = callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)

reduce_lr = callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=0.00001

)

checkpoint = callbacks.ModelCheckpoint(

    "best_cat_mood_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"

)

# MODEL SUMMARY
print("\n")
model.summary()

# TRAIN
print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],
    verbose=1

)

# TEST MODEL
print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(

    X_test,
    y_test,
    verbose=1
)

print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy (%): "
    f"{test_accuracy * 100:.2f}%"
)

# SAVE FINAL MODEL
model.save(
    "cat_mood_model.keras"
)
print(
    "\nModel saved as "
    "'cat_mood_model.keras'"
)
print("=" * 60)