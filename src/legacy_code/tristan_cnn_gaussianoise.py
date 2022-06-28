import tensorflow as tf
# from tensorflow.keras import models, layers
import tensorflow_datasets as tfds
import tensorflow_io as tfio
# import tfplot
import matplotlib.pyplot as plt
# from IPython.display import Audio, display
import numpy as np
# import librosa
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from keras.preprocessing.image import ImageDataGenerator


def process(sample):
    audio = sample["audio"]
    label = sample["label"]

    audio = tf.cast(audio, tf.float32) / 32768.0

    spectrogram = tfio.audio.spectrogram(
        audio, nfft=1024, window=1024, stride=64
    )

    spectrogram = tfio.audio.melscale(
        spectrogram, rate=8000, mels=64, fmin=0, fmax=2000
    )
    spectrogram /= tf.math.reduce_max(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.image.resize(spectrogram, (64, 64))
    spectrogram = tf.transpose(spectrogram, perm=(1, 0, 2))
    spectrogram = spectrogram[::-1, :, :]

    return spectrogram, label


print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

(dataset_train_original, dataset_validate_original), info = tfds.load(
    "spoken_digit",
    split=["train[:80%]", "train[80%:]"],
    with_info=True
)

# print(dataset_validate_original.take(1))
# print(info)

# for sample in dataset_train_original.shuffle(2500).take(1):
#     audio = sample["audio"].numpy().astype("float32")
#     label = sample["label"].numpy()
#
#     plt.plot(audio)
#     plt.show()
#     plt.close()
#
#     mel = librosa.feature.melspectrogram(
#         y=audio,
#         sr=8000,
#         n_mels=64,
#         hop_length=64,
#         fmax=2000
#     )
#
#     mel /= np.max(mel)
#     print(mel.shape)
#     plt.imshow(mel[::-1, :], cmap="inferno")
#     plt.title(f"label: {label}")
#     plt.savefig('mel.png')
#
#     # display(Audio(audio, rate=8000))


dataset = dataset_train_original.map(lambda sample: process(sample))
#
print(dataset.cardinality().numpy())
for element in dataset:
  print(element)
# for x, y, in dataset:
#     x = x.numpy()
#     plt.imshow(x.squeeze(), cmap="inferno")
#     plt.savefig("new_mel.png")
#
# dataset_train = dataset_train_original.map(lambda sample: process(sample))
# dataset_train = dataset_train.cache()
# dataset_train = dataset_train.shuffle(2500)
# dataset_train = dataset_train.batch(32)
#
# dataset_validate = dataset_validate_original.map(lambda sample: process(sample))
# dataset_validate = dataset_validate.cache()
# dataset_validate = dataset_validate.batch(32)
#
# epochs = 2000
#
# model = tf.keras.models.Sequential()
#
# model.add(tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu", input_shape=(64, 64, 1)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# # model.add(tf.keras.layers.Dropout(0.2))
#
# model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# # model.add(tf.keras.layers.Dropout(0.2))
#
# model.add(tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# # model.add(tf.keras.layers.Dropout(0.3))
#
# model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# # model.add(tf.keras.layers.Dropout(0.4))
#
# model.add(tf.keras.layers.Flatten())
# # model.add(tf.keras.layers.Dropout(0.3))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.GaussianNoise(0.05))
# model.add(tf.keras.layers.Dense(256, activation="relu"))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.GaussianNoise(0.1))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.GaussianNoise(0.2))
# model.add(tf.keras.layers.Dense(64, activation="relu"))
# model.add(tf.keras.layers.Dropout(0.6))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))
# model.summary()
#
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
#
# history = model.fit(
#     dataset_train,
#     epochs=epochs,
#     validation_data=dataset_validate
# )
#
# plt.plot(history.history["loss"], label="loss")
# plt.plot(history.history["val_loss"], label="val_loss")
# plt.legend()
# plt.show()
# plt.savefig("history1.png")
# plt.close()
#
# plt.plot(history.history["accuracy"], label="accuracy")
# plt.plot(history.history["val_accuracy"], label="val_accuracy")
# plt.legend()
# plt.show()
# plt.savefig("history2.png")
# plt.close()