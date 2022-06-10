import tensorflow as tf
import tensorflow_io as tfio
import glob
import soundfile
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm


dataset = tf.data.experimental.load("pickled_distance_dataset.dat")
dataset = dataset.shuffle(72000)

dataset_train = dataset.take(57600)
dataset_validate = dataset.skip(57600)

dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(57600)
dataset_train = dataset_train.batch(32)

dataset_validate = dataset_validate.cache()
dataset_validate = dataset_validate.batch(32)

# epochs = 2000
epochs = 50

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
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.GaussianNoise(0.05))
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.GaussianNoise(0.1))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.GaussianNoise(0.2))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.6))
model.add(tf.keras.layers.Dense(2, activation="softmax"))
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_validate
)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()
plt.savefig("history1_silly_distance.png")
plt.close()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()
plt.savefig("history2_silly_distance.png")
plt.close()


model.save('gaussian_noise_dropout_distance.model')