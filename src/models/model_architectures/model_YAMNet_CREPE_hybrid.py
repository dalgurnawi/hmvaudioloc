import crepe
import tensorflow as tf


def create_yamnet_crepe_model():
    crepe_model = crepe.core.build_and_load_model("full")
    crepe_model.layers.pop()
    crepe_model = tf.keras.models.Model(crepe_model.inputs, crepe_model.layers[-2].output)
    crepe_model.trainable = False
    crepe_model.summary()

    input_shape = (1024, 2)
    x = tf.keras.layers.Input(shape=input_shape)

    y_left = tf.keras.layers.Lambda(lambda x: x[:, :, 0], output_shape=(1024,))(x)
    embedded_left = crepe_model(y_left)

    y_right = tf.keras.layers.Lambda(lambda x: x[:, :, 1], output_shape=(1024,))(x)
    embedded_right = crepe_model(y_right)

    comb_out = tf.keras.layers.Concatenate()([embedded_left, embedded_right])

    z = tf.keras.layers.Dense(4096, activation="relu")(comb_out)
    z = tf.keras.layers.Dense(1024, activation="relu")(z)
    z = tf.keras.layers.Dense(512, activation="relu")(z)
    z = tf.keras.layers.Dropout(0.5)(z)
    z = tf.keras.layers.GaussianNoise(0.05)(z)
    z = tf.keras.layers.Dense(256, activation="relu")(z)
    z = tf.keras.layers.Dropout(0.5)(z)
    z = tf.keras.layers.GaussianNoise(0.1)(z)
    z = tf.keras.layers.Dense(128, activation="relu")(z)
    z = tf.keras.layers.Dropout(0.5)(z)
    z = tf.keras.layers.GaussianNoise(0.2)(z)
    z = tf.keras.layers.Dense(64, activation="relu")(z)
    z = tf.keras.layers.Dropout(0.6)(z)
    output_dense = tf.keras.layers.Dense(8, activation="softmax")(z)
    # output_dense = tf.keras.layers.Dense(8, activation="softmax")(comb_out)
    model = tf.keras.models.Model(x, output_dense)

    return model

