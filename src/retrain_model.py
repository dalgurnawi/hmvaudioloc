import tensorflow as tf
import tensorflow_io as tfio
import glob
# import soundfile
import matplotlib.pyplot as plt
import numpy as np
# import pickle
from tqdm import tqdm
import random
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def map_random_sample_file(test_folder_name):
    list_of_all_files = glob.glob(f'{test_folder_name}/*.wav')
    file_name = list_of_all_files[random.randint(0, len(list_of_all_files))]
    print(file_name)
    # just one:
    # for file_name in [list_of_all_files[0]]:
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
    return np.array([full_spectrogram]), np.array([label])


def map_all_test_files(test_folder_name):
    list_of_all_files = glob.glob(f'{test_folder_name}/*.wav')
    list_of_spectrograms = []
    list_of_labels = []
    for file_name in tqdm(list_of_all_files):
        audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
        audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
        spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
        spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
        list_of_spectrograms.append(tf.concat([spectrogram_1, spectrogram_2], axis=-1))
        list_of_labels.append(tf.convert_to_tensor(int(file_name.split("_")[-1][:-4])))
    return np.array(list_of_spectrograms), np.array(list_of_labels)


epochs = 200

dataset = tf.data.experimental.load("pickled_real_world_dataset.dat")
dataset = dataset.shuffle(24000)

dataset_train = dataset.take(21600)
dataset_validate = dataset.skip(21600)

dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(21600)
dataset_train = dataset_train.batch(32)

dataset_validate = dataset_validate.cache()
dataset_validate = dataset_validate.batch(32)

# print(test_input)
# print(test_target)

reconstructed_model = tf.keras.models.load_model('gaussian_noise_dropout.model')

history = reconstructed_model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_validate
)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.savefig("history1_retrain_dropout_Gnoise_real_world.png")
plt.close()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.savefig("history2_retrain_dropout_Gnoise_real_world.png")
plt.close()

tf.keras.models.save_model(reconstructed_model, 'real_world_retrained_gnoise_dropout.model')