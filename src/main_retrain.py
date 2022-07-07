from src.utils.google_drive_download_datasets import download_real_world_spoken_mnist_dataset
from src.utils.google_drive_download_datasets import download_real_world_complex_dataset

from src.data.read_pickled_dataset import read_pickled_dataset

from src.models.model_retrain import retrain_model

from src.models.model_test import test_all_trained_models


download_real_world_spoken_mnist_dataset()
download_real_world_complex_dataset()

(real_world_spoken_mnist_dataset_train, real_world_spoken_mnist_dataset_validate) = \
    read_pickled_dataset(pickled_dataset_relative_path="../../data/datasets/pickled_real_world_spoken_mnist_dataset.dat",
                         dataset_size=72000)

# retrain_all_models(dataset_train, dataset_validate, epochs=10)
# test_all_trained_models(after_retraining_specific_suffix=.rl_mnist_e10)

(real_world_complex_dataset_train, real_world_complex_dataset_validate) = \
    read_pickled_dataset(pickled_dataset_relative_path="../../data/datasets/pickled_real_world_spoken_mnist_dataset.dat",
                         dataset_size=72000)

# retrain_all_models(dataset_train, dataset_validate, epochs=10)
# test_all_trained_models(after_retraining_specific_suffix=.rl_mnist_e10)

