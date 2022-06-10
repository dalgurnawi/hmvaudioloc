import tensorflow as tf
import tensorflow_io as tfio
import glob
import soundfile
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import time

tf.compat.v1.enable_eager_execution()


def process(audio):
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

    return spectrogram


def map_dataset():
    list_of_all_files = glob.glob('temp_small_data/*.wav')

    # just one:
    # for file_name in [list_of_all_files[0]]:
    for file_name in tqdm(list_of_all_files):
        # # if convert_to_16bit:
        # #     file_name = 'temp.wav'
        # #     data, samplerate = soundfile.read(file_name)
        # #     soundfile.write(file_name, data, samplerate, subtype='PCM_16')
        audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
        audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
        spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
        spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
        full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
        label = tf.convert_to_tensor(int(file_name.split("_")[-1][:-4]))
        yield full_spectrogram, label


# map_dataset()
# list_of_all_files = glob.glob('data_16/*/*.wav')
# print(len(list_of_all_files))
# dataset_mapper = make_map_callable(list_of_all_files[0])
# print(dataset_mapper)

dataset = tf.data.Dataset.from_generator(map_dataset, output_signature=(tf.TensorSpec(shape=(64, 64, 2), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(), dtype=tf.int32)))
dataset.shuffle(72)
print("create and shuffle")

start_time = time.time()
tf.data.experimental.save(dataset, "pickled_dataset_temp_small.dat")
stop_time = time.time()
print(f"The pickling of 72 samples took {stop_time - start_time}s")
# because datasetmap is flat??!?!? it has no cardinality so You cant get the total size of dataset
# dataset.cardinality().numpy()

# print singular entries of dataset... SO IT IS WORKING?!?!?!?!!
# for singular_data in dataset:
#     print(singular_data)


# total_num_samples = 72000
#
# train_size = int(0.7 * total_num_samples)
# val_size = int(0.15 * total_num_samples)
# test_size = int(0.15 * total_num_samples)
#
# dataset = dataset.shuffle()
# train_dataset = dataset.take(train_size)
# test_dataset = dataset.skip(train_size)
# val_dataset = test_dataset.skip(val_size)
# test_dataset = test_dataset.take(test_size)


# This should work for all files in specified folder, but its a lot
# list_of_all_files = glob.glob('data_16/*/*.wav')
# total_num_samples = len(list_of_all_files)
# train_num = total_num_samples - (total_num_samples * 0.2)
# dataset_train = dataset.take(train_num)
# dataset_validate = dataset.skip(train_num)
#
# dataset_train = dataset_train.cache()
# dataset_train = dataset_train.shuffle(train_num)

# dataset_train = dataset.take(57600)
# # dataset_validate = dataset.skip(14400)
# dataset_validate = dataset.skip(57600)








# dataset.shuffle(72000)
#
# dataset_train = dataset.take(57600)
# dataset_validate = dataset.skip(57600)
#
# dataset_train = dataset_train.cache()
# dataset_train = dataset_train.shuffle(57600)
# dataset_train = dataset_train.batch(32)
#
# dataset_validate = dataset_validate.cache()
# dataset_validate = dataset_validate.batch(32)
#
# # epochs = 2000
# epochs = 200
#
# model = tf.keras.models.Sequential()
#
# model.add(tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu", input_shape=(64, 64, 2)))
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
# model.add(tf.keras.layers.Dense(8, activation="softmax"))
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














# for element in dataset:
#     print(element)


# file_name = glob.glob('all_outputs/0_george_0_45down_sector_0.wav')
# # file_name = glob.glob('data_16/sector_0/0_george_0_45down_sector_0.wav')
#
# print(file_name)
# test_file = tf.io.read_file(file_name[0])
#


# file_name = 'data_16/sector_0/0_george_4_45up_sector_0.wav'
# audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
# print(audio_tensor)
# t1, t2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
# print(t1.numpy().squeeze().astype("float32"))
# print(t2.numpy().squeeze().astype("float32"))

# plt.plot(t1.numpy().astype("float32"))
# test_audio, _ = tf.audio.decode_wav(contents='new.wav')
# print(test_audio.shape)

# spect = process(t1.numpy().squeeze().astype("float32"))
# spect2 = process(t2.numpy().squeeze().astype("float32"))
# full_input = tf.concat([spect, spect2], axis=-1)
# category_name = file_name[0].split("_")[-1][:-4]
# print(category_name)
# print(full_input)

# pair = tf.data.Dataset.map([full_input, tf.convert_to_tensor(float(category_name))])
# print(pair)
# dataset_list.append([full_input, float(category_name)])
#
# dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
# print(dataset)

