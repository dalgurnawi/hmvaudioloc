import tensorflow as tf


def create_baseline_model():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu", input_shape=(64, 64, 2)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.GaussianNoise(0.05))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.GaussianNoise(0.1))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.GaussianNoise(0.2))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    # model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(8, activation="softmax"))

    return model

