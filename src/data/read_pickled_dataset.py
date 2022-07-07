import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_pickled_dataset(pickled_dataset_relative_path, dataset_size, validation_split=0.2, batch_size=32):
    dirname = os.path.dirname(__file__)

    dataset = tf.data.experimental.load(os.path.join(dirname, pickled_dataset_relative_path))
    dataset = dataset.shuffle(dataset_size)
    train_split = dataset_size - (dataset_size * validation_split)
    dataset_train = dataset.take(train_split)
    dataset_validate = dataset.skip(train_split)

    dataset_train = dataset_train.cache()
    dataset_train = dataset_train.shuffle(train_split)
    dataset_train = dataset_train.batch(batch_size)

    dataset_validate = dataset_validate.cache()
    dataset_validate = dataset_validate.batch(batch_size)

    return dataset_train, dataset_validate