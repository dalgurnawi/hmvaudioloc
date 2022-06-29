import tensorflow as tf


def create_baseline_vanilla_model():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(4, (3, 3), activation="relu", padding= "same", input_shape=(64, 64, 2)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="softmax"))

    return model

