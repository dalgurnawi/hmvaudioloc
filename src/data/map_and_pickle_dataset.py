import tensorflow as tf
import tensorflow_io as tfio
import glob
import soundfile
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

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
    list_of_all_files = glob.glob('data_16/*/*.wav')

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
dataset.shuffle(72000)
print("create and shuffle")

tf.data.experimental.save(dataset, "pickled_dataset.dat")
print("pickling complete")