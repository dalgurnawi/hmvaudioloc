from src.utils.google_drive_download_datasets import download_augmented_spoken_mnist_dataset
from src.utils.google_drive_download_datasets import download_real_world_spoken_test_dataset

from src.data.map_and_pickle_dataset import map_augmented_spoken_mnist_dataset
from src.data.map_and_pickle_dataset import map_real_world_spoken_mnist_dataset
from src.data.map_and_pickle_dataset import map_real_world_complex_dataset
from src.data.map_and_pickle_dataset import map_real_world_spoken_test_dataset

from src.models.model_train import train_model_plot_and_save

from src.models.model_test import test_all_trained_models

from src.data.read_pickled_dataset import read_pickled_dataset


download_augmented_spoken_mnist_dataset()
download_real_world_spoken_test_dataset()

map_augmented_spoken_mnist_dataset()
map_real_world_spoken_mnist_dataset()
map_real_world_complex_dataset()
map_real_world_spoken_test_dataset()

(augmented_spoken_mnist_dataset_train, augmented_spoken_mnist_dataset_validate) = \
    read_pickled_dataset(pickled_dataset_relative_path="../../data/datasets/pickled_augmented_spoken_mnist_dataset.dat",
                         dataset_size=72000)

train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="baseline_model")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_1")
# train_model_plot_and_save(audmented_spoken_mnist_dataset_train,
# augmented_spoken_mnist_dataset_validate,
# model_choice="model_paper_code_2")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_3")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_4")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_5")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_6")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_7")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_8")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_9")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_10")
train_model_plot_and_save(augmented_spoken_mnist_dataset_train,
                          augmented_spoken_mnist_dataset_validate,
                          model_choice="model_paper_code_11")

test_all_trained_models()

