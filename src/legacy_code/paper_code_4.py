import tensorflow as tf
from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input
from keras.models import Model
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

# An overview of the paper's SELD-TCN architecture

# Model Architecture Two :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_seldtcn_model(dropout_rate):

    # model definition
    spec_start =  Input(shape=(1, 64, 64))
    spec_cnn = spec_start
    spec_cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1,2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,32), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(2,4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1,4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(2,2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1,4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,2), padding = "same")(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1,4), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Activation("relu")(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    
    doa = tf.keras.layers.Flatten()(spec_cnn)
    doa = tf.keras.layers.Dense(512)(doa)
    doa = tf.keras.layers.Activation('relu')(doa)
    doa = tf.keras.layers.BatchNormalization()(doa)
    doa = tf.keras.layers.Dropout(dropout_rate)(doa)
    doa = tf.keras.layers.Dense(12,activation='softmax')(doa)

    model = Model(inputs=spec_start, outputs=doa)
    model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics='accuracy')

    model.summary()
    return model


model = get_seldtcn_model(dropout_rate=0.5)

epochs = 10
history = model.fit(dataset_train,
                    epochs=epochs
                    ,validation_data=dataset_validate)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.savefig("history1_paper_code_4.png")
plt.close()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.savefig("history2_paper_code_4.png")
plt.close()

model.save('paper_code_4.model')