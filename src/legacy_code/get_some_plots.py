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


def map_sample_file(file_name):
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
    # full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
    # label = tf.convert_to_tensor(int(file_name.split("_")[-1][:-4]))

    return spectrogram_1, spectrogram_2


file_shard = "4_jackson_42_"
list_of_all_files = sorted(glob.glob(f'data_16/*/{file_shard}*.wav'))

fig, axs = plt.subplots(8, 3, figsize=(27, 27))
fig.suptitle(f'All the plots all the time of: {file_shard}')
file_list_index = 0
for x in range(8):
    for y in range(3):
        s1, s2 = map_sample_file(list_of_all_files[file_list_index])
        double_spectrogram = np.concatenate((s1, s2), axis=1)
        axs[x, y].imshow(double_spectrogram[::-1, :], cmap="inferno")
        axs[x, y].set_title(f'{file_shard} from Sector:{list_of_all_files[file_list_index].split("_")[-1][:-4]}, elevation: {list_of_all_files[file_list_index].split("_")[-3]}')
        file_list_index += 1
plt.savefig(f'{file_shard}super_plot.png')

    # s1, s2 = map_sample_file(afile)
    # ax1.plot(s1[::-1, :], cmap="inferno")
    # ax2.plot(s1[::-1, :], cmap="inferno")
    # break


#
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')