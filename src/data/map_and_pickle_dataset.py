import tensorflow as tf
import tensorflow_io as tfio
import glob
from tqdm import tqdm
import os

tf.compat.v1.enable_eager_execution()
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


def map_dataset_generator_base():
    list_of_all_files = glob.glob('direct-path-to-folder')
    for file_name in tqdm(list_of_all_files):
        audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
        audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
        spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
        spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
        full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
        label = tf.convert_to_tensor(int(file_name.split("_")[-1][:-4]))

        yield full_spectrogram, label


def map_dataset_base(samples_folder):
    dirname = os.path.dirname(__file__)
    folder_name = samples_folder.split('/')[-1]

    dataset = tf.data.Dataset.from_generator(map_dataset_generator_base,
                                             output_signature=(tf.TensorSpec(shape=(64, 64, 2),
                                                                             dtype=tf.float32),
                                                               tf.TensorSpec(shape=(),
                                                                             dtype=tf.int32)))
    dataset.shuffle(72000)
    tf.data.experimental.save(dataset, os.path.join(dirname, f"{folder_name}.dat"))


def map_augmented_spoken_mnist_dataset():
    dirname = os.path.dirname(__file__)
    samples_folder_path = os.path.join(dirname, '../../data/example_recordings/augmented_spoken_mnist_data')
    samples_folder_name = samples_folder_path.split('/')[-1]

    dataset = tf.data.Dataset.from_generator(map_dataset_generator_base,
                                             output_signature=(tf.TensorSpec(shape=(64, 64, 2),
                                                                             dtype=tf.float32),
                                                               tf.TensorSpec(shape=(),
                                                                             dtype=tf.int32)))
    dataset.shuffle(72000)
    tf.data.experimental.save(dataset, os.path.join(dirname, f"{samples_folder_name}.dat"))


def map_real_world_spoken_mnist_dataset():
    dirname = os.path.dirname(__file__)
    samples_folder_path = os.path.join(dirname, '../../data/example_recordings/real_world_spoken_mnist_data')
    samples_folder_name = samples_folder_path.split('/')[-1]

    dataset = tf.data.Dataset.from_generator(map_dataset_generator_base,
                                             output_signature=(tf.TensorSpec(shape=(64, 64, 2),
                                                                             dtype=tf.float32),
                                                               tf.TensorSpec(shape=(),
                                                                             dtype=tf.int32)))
    dataset.shuffle(72000)
    tf.data.experimental.save(dataset, os.path.join(dirname, f"{samples_folder_name}.dat"))


def map_real_world_complex_dataset():
    dirname = os.path.dirname(__file__)
    samples_folder_path = os.path.join(dirname, '../../data/example_recordings/real_world_complex_data')
    samples_folder_name = samples_folder_path.split('/')[-1]

    dataset = tf.data.Dataset.from_generator(map_dataset_generator_base,
                                             output_signature=(tf.TensorSpec(shape=(64, 64, 2),
                                                                             dtype=tf.float32),
                                                               tf.TensorSpec(shape=(),
                                                                             dtype=tf.int32)))
    dataset.shuffle(72000)
    tf.data.experimental.save(dataset, os.path.join(dirname, f"{samples_folder_name}.dat"))


def map_real_world_spoken_test_dataset():
    dirname = os.path.dirname(__file__)
    samples_folder_path = os.path.join(dirname, '../../data/example_recordings/real_world_spoken_test_data')
    samples_folder_name = samples_folder_path.split('/')[-1]

    dataset = tf.data.Dataset.from_generator(map_dataset_generator_base,
                                             output_signature=(tf.TensorSpec(shape=(64, 64, 2),
                                                                             dtype=tf.float32),
                                                               tf.TensorSpec(shape=(),
                                                                             dtype=tf.int32)))
    dataset.shuffle(72000)
    tf.data.experimental.save(dataset, os.path.join(dirname, f"{samples_folder_name}.dat"))


def map_augmented_mnist_dataset_generator():
    dirname = os.path.dirname(__file__)
    list_of_all_files = glob.glob(os.path.join(dirname, '../../data/example_recordings/augmented_mnist_data'))
    for file_name in tqdm(list_of_all_files):
        audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
        audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
        spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
        spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
        full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
        label = tf.convert_to_tensor(int(file_name.split("_")[-1][:-4]))
        yield full_spectrogram, label


def map_real_world_mnist_dataset_generator():
    dirname = os.path.dirname(__file__)
    list_of_all_files = glob.glob(os.path.join(dirname, '../../data/example_recordings/real_world_mnist_data'))
    for file_name in tqdm(list_of_all_files):
        audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
        audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
        spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
        spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
        full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
        label = tf.convert_to_tensor(int(file_name.split("_")[-1][:-4]))
        yield full_spectrogram, label


def map_real_world_complex_dataset_generator():
    dirname = os.path.dirname(__file__)
    list_of_all_files = glob.glob(os.path.join(dirname, '../../data/example_recordings/real_world_complex_data'))
    for file_name in tqdm(list_of_all_files):
        audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
        audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
        spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
        spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
        full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
        label = tf.convert_to_tensor(int(file_name.split("_")[-1][:-4]))
        yield full_spectrogram, label


def map_real_world_spoken_test_dataset_generator():
    dirname = os.path.dirname(__file__)
    list_of_all_files = glob.glob(os.path.join(dirname, '../../data/example_recordings/real_world_spoken_test_data'))
    for file_name in tqdm(list_of_all_files):
        audio_tensor = tfio.audio.AudioIOTensor(file_name).to_tensor()
        audio_channel_1, audio_channel_2 = tf.split(audio_tensor, num_or_size_splits=2, axis=1)
        spectrogram_1 = process(audio_channel_1.numpy().squeeze().astype("float32"))
        spectrogram_2 = process(audio_channel_2.numpy().squeeze().astype("float32"))
        full_spectrogram = tf.concat([spectrogram_1, spectrogram_2], axis=-1)
        label = tf.convert_to_tensor(int(file_name.split("_")[-1][:-4]))
        yield full_spectrogram, label
