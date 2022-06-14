from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = tf.data.experimental.load("pickled_dataset.dat")
dataset = dataset.shuffle(72000)

dataset_train = dataset.take(57600)
dataset_validate = dataset.skip(57600)

dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(57600)
dataset_train = dataset_train.batch(32)

dataset_validate = dataset_validate.cache()
dataset_validate = dataset_validate.batch(32)


def get_seldtcn_model(dropout_rate):
    # model definition
    # spec_start = tf.keras.Sequential()
    spec_start = Input(shape=(64, 64, 2))
    spec_cnn = spec_start
    spec_cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 16), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 16), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    doa = tf.keras.layers.Flatten()(spec_cnn)
    doa = tf.keras.layers.Dense(512)(doa)
    doa = tf.keras.layers.Activation('relu')(doa)
    doa = tf.keras.layers.BatchNormalization()(doa)
    doa = tf.keras.layers.Dropout(dropout_rate)(doa)
    doa = tf.keras.layers.Dense(8, activation='softmax')(doa)
    model = Model(inputs=spec_start, outputs=doa)
    model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.summary()
    return model


sequence_length = 512,  # Feature sequence length
batch_size = 16,  # Batch size
dropout_rate = 0.0,  # Dropout rate, constant for all layers
nb_cnn2d_filt = 64,  # Number of CNN nodes, constant for each layer
pool_size = [8, 8, 2],  # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
rnn_size = [128, 128],  # RNN contents, length of list = number of layers, list value = number of nodes
fnn_size = [128],  # FNN contents, length of list = number of layers, list value = number of nodes
loss_weights = [1., 50.],  # [sed, doa] weight for scaling the DNN outputs
xyz_def_zero = True,  # Use default DOA Cartesian value x,y,z = 0,0,0
nb_epochs = 500,  # Train for maximum epochs
model = get_seldtcn_model(dropout_rate=0.5)
epochs = 10

history = model.fit(dataset_train,
                    epochs=epochs,
                    validation_data=dataset_validate)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.savefig("history1_paper_code_1.png")
plt.close()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.savefig("history2_paper_code_1.png")
plt.close()

model.save('paper_code_1.model')