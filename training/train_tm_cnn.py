import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Models paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(ROOT, "datasets", "classification", "train")
TEST_DIR = os.path.join(ROOT, "datasets", "classification", "test")
OUTPUT_MODEL = os.path.join(ROOT, "models", "cnn", "tm_cnn.keras")

# Training parameters

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 100
NUM_CLASSES = 3

# Data augmentation

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Datasets

train_ds = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training"
)

val_ds = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation"
)

test_ds = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

print("\nCLASS INDICES:")
print(train_ds.class_indices)

# Base model

base_model = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False

# Model

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# Compile model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1
    )
]

# Train model

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
)

# Evaluation - test model

print("\nTESTING MODEL...\n")
loss, accuracy = model.evaluate(test_ds)
print(f"\nTEST ACCURACY: {accuracy:.4f}")

# Save model

os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
model.save(OUTPUT_MODEL)

print("\nMODEL SAVED:")
print(OUTPUT_MODEL)
