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


test_folder_name = "real_world_level_sector_0"
# test_input, test_target = map_random_sample_file()
test_input, test_target = map_all_test_files(test_folder_name)

# print(test_input)
# print(test_target)

reconstructed_model = tf.keras.models.load_model('gaussian_noise_dropout.model')

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
start_time = time.time()
prediction = reconstructed_model.predict(test_input)
print("predictions shape:", prediction.shape)
prediction_probabilities = tf.math.top_k(prediction, k=1)
print(prediction_probabilities)
top_1_scores = prediction_probabilities.values.numpy()
print(top_1_scores)
dict_class_entries = prediction_probabilities.indices.numpy()
plt.hist(dict_class_entries)
plt.title(f"{test_folder_name}")
plt.savefig(f"{test_folder_name}.png")
print(dict_class_entries)
end_time = time.time()
print(f"procedure done in {end_time - start_time}s")
