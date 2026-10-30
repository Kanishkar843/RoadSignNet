"""Compact CNN adapted from Vishal Kumar's MIT-licensed GTSRB baseline."""

import tensorflow as tf


def configure_runtime(seed: int = 42):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # TensorFlow may already be initialized by the hosting application.
    tf.config.experimental.enable_op_determinism()


def build_model(augment: bool = True):
    layers = [tf.keras.layers.Input(shape=(32, 32, 3))]
    if augment:
        layers += [
            tf.keras.layers.RandomRotation(0.025, fill_mode="reflect", seed=42),
            tf.keras.layers.RandomTranslation(0.05, 0.05, seed=43),
            tf.keras.layers.RandomZoom(0.05, seed=44),
            tf.keras.layers.RandomBrightness(0.1, value_range=(0, 1), seed=45),
        ]
    layers += [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(43, activation="softmax"),
    ]
    model = tf.keras.Sequential(layers, name="RoadSignNet")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def dataset(images, labels, batch_size: int, training: bool = False, seed: int = 42):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(labels), seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return ds.batch(batch_size).with_options(options).prefetch(1)
