import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2

from sklearn.utils.class_weight import compute_class_weight


# SETTINGS
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 20
BATCH_SIZE = 8
INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00001

IMG_SIZE = 224

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
print("CAT MOOD TRANSFER LEARNING")
print("=" * 60)
print(
    f"\nTraining data:   {X_train.shape}"
)

print(
    f"Validation data: {X_val.shape}"
)

print(
    f"Testing data:    {X_test.shape}"
)

print(
    f"Classes: {len(classes)}"
)

print("\nClass names:")

for i, class_name in enumerate(classes):

    print(
        f"{i}: {class_name}"
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

    # Resize to MobileNetV2 input
    X = tf.image.resize(
        X,
        [IMG_SIZE, IMG_SIZE]
    )

    # Convert grayscale to RGB
    X = tf.image.grayscale_to_rgb(
        X
    )
    return X.numpy()

print(
    "\nPreparing spectrograms "
    "for MobileNetV2..."
)
X_train = prepare_spectrograms(
    X_train
)
X_val = prepare_spectrograms(
    X_val
)
X_test = prepare_spectrograms(
    X_test
)

print(
    f"Prepared training data: "
    f"{X_train.shape}"
)

print(
    f"Prepared validation data: "
    f"{X_val.shape}"
)

print(
    f"Prepared testing data: "
    f"{X_test.shape}"
)

# CLASS WEIGHTS
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=y_train
)

class_weights = {
    i: weight
    for i, weight in enumerate(
        class_weights_array
    )
}

print(
    "\nCLASS WEIGHTS"
)

print("-" * 60)

for i, class_name in enumerate(classes):
    print(
        f"{class_name:<15} "
        f"{class_weights[i]:.3f}"
    )

# DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10
    ),

    layers.RandomRotation(
        0.03
    ),

], name="spectrogram_augmentation")

# MOBILE NET V2
base_model = MobileNetV2(

    weights="imagenet",
    include_top=False,
    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# Freeze entire backbone initially
base_model.trainable = False

# BUILD MODEL
inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)


# Data augmentation
x = data_augmentation(
    inputs
)


# MobileNetV2
x = base_model(
    x,
    training=False
)


# Global feature extraction
x = layers.GlobalAveragePooling2D()(x)


# Classifier
x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(
    0.5
)(x)

outputs = layers.Dense(
    len(classes),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# COMPILE INITIAL MODEL
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

# CALLBACKS
checkpoint = callbacks.ModelCheckpoint(

    "best_cat_mood_transfer_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stopping = callbacks.EarlyStopping(

    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = callbacks.ReduceLROnPlateau(

    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)


# MODEL SUMMARY
print("\n")

model.summary()

# INITIAL TRAINING
print("\n")
print("=" * 60)
print("STAGE 1: TRAINING CLASSIFIER")
print("=" * 60)


history_initial = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),

    epochs=INITIAL_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# LOAD BEST CLASSIFIER
model = tf.keras.models.load_model(
    "best_cat_mood_transfer_model.keras"
)

# FINE-TUNING
print("\n")
print("=" * 60)
print("STAGE 2: FINE-TUNING MOBILENETV2")
print("=" * 60)


# Unfreeze MobileNetV2
base_model = model.get_layer(
    "mobilenetv2_1.00_224"
)

base_model.trainable = True


# Keep early layers frozen.
# Only the last 20 layers are allowed
# to adapt to cat sounds.

for layer in base_model.layers[:-20]:
    layer.trainable = False


# Keep BatchNormalization layers frozen.
# This helps prevent unstable updates
# with the small dataset.

for layer in base_model.layers:
    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False

print(
    "\nTrainable MobileNetV2 layers:"
)

trainable_count = 0

for layer in base_model.layers:
    if layer.trainable:

        trainable_count += 1


print(
    trainable_count
)

# COMPILE FINE-TUNING MODEL
model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)

# FINE-TUNING CALLBACKS
fine_tune_checkpoint = callbacks.ModelCheckpoint(
    "best_cat_mood_transfer_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

fine_tune_early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)


fine_tune_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

# FINE-TUNE
history_fine = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),

    epochs=FINE_TUNE_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[
        fine_tune_checkpoint,
        fine_tune_early_stopping,
        fine_tune_reduce_lr
    ],
    verbose=1
)

# LOAD BEST MODEL
model = tf.keras.models.load_model(
    "best_cat_mood_transfer_model.keras"
)

# FINAL TEST RESULTS
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
    "cat_mood_transfer_model.keras"
)

print(
    "\nModel saved as "
    "'cat_mood_transfer_model.keras'"
)

print("=" * 60)