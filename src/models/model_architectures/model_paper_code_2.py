import tensorflow as tf
from keras.layers import Input
from keras.models import Model


def create_paper_code_2_model(dropout_rate=0.5):
    spec_start = Input(shape=(64, 64, 2))
    spec_cnn = spec_start

    # CONVOLUTIONAL LAYERS =========================================================
    # First convolutional layer
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Dropout(0)(spec_cnn)

    # Second convolutional layer
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Dropout(0)(spec_cnn)

    # Third convolutional layer
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Dropout(0)(spec_cnn)

    # resblock_input = Permute((2, 1, 3))(spec_cnn)
    # resblock_input = Reshape((64, -1))(spec_cnn)
    resblock_input = spec_cnn

    # TCN layer ===================================================================

    # residual blocks ------------------------
    skip_connections = []

    for d in range(8):

        # 2D convolution
        spec_conv2d = tf.keras.layers.Convolution2D(filters=256,
                                                    kernel_size=(3, 3),
                                                    padding='same',
                                                    dilation_rate=2 ** d)(resblock_input)
        spec_conv2d = tf.keras.layers.BatchNormalization()(spec_conv2d)

        # activations
        tanh_out = tf.keras.layers.Activation('tanh')(spec_conv2d)
        sigm_out = tf.keras.layers.Activation('sigmoid')(spec_conv2d)
        spec_act = tf.keras.layers.Multiply()([tanh_out, sigm_out])

        # spatial dropout
        spec_drop = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)(spec_act)

        # 2D convolution
        skip_output = tf.keras.layers.Convolution2D(filters=128,
                                                    kernel_size=(1, 1),
                                                    padding='same')(spec_drop)

        res_output = tf.keras.layers.Add()([resblock_input, skip_output])

        if skip_output is not None:
            skip_connections.append(skip_output)

        resblock_input = res_output
    # ---------------------------------------

    # Residual blocks sum
    spec_sum = tf.keras.layers.Add()(skip_connections)
    spec_sum = tf.keras.layers.Activation('relu')(spec_sum)

    # 2D convolution
    spec_conv2d_2 = tf.keras.layers.Convolution2D(filters=128,
                                                  kernel_size=(1, 1),
                                                  padding='same')(spec_sum)
    spec_conv2d_2 = tf.keras.layers.Activation('relu')(spec_conv2d_2)

    # 2D convolution
    spec_tcn = tf.keras.layers.Convolution2D(filters=128,
                                             kernel_size=(1, 1),
                                             padding='same')(spec_conv2d_2)
    spec_tcn = tf.keras.layers.Activation('tanh')(spec_tcn)

    # Output ==================================================================
    # doa = spec_tcn
    # doa = tf.keras.layers.Activation('tanh')(spec_tcn)
    # doa = tf.keras.layers.Activation('softmax', name='doa_out')(doa)

    doa = tf.keras.layers.Flatten()(spec_tcn)
    # doa = tf.keras.layers.Dense(64, activation="sigmoid")(doa)
    doa = tf.keras.layers.Dense(8, activation="softmax")(doa)

    model = Model(inputs=spec_start, outputs=doa)

    return model

