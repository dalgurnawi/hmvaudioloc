import tensorflow as tf
from keras.layers import Input
from keras.models import Model


def create_paper_code_4_model(dropout_rate=0.5):
    spec_start = Input(shape=(64, 64, 2))
    spec_cnn = spec_start
    spec_cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 32), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)

    doa = tf.keras.layers.Flatten()(spec_cnn)
    doa = tf.keras.layers.Dense(512)(doa)
    doa = tf.keras.layers.Activation('relu')(doa)
    doa = tf.keras.layers.BatchNormalization()(doa)
    doa = tf.keras.layers.Dropout(dropout_rate)(doa)
    doa = tf.keras.layers.Dense(8, activation='softmax')(doa)

    model = Model(inputs=spec_start, outputs=doa)
    # model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics='accuracy')
    #
    # model.summary()
    return model

