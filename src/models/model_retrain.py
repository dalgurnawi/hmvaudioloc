import tensorflow as tf
import matplotlib.pyplot as plt
import os

from src.data.read_pickled_dataset import read_pickled_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def retrain_model(pickled_model_relative_path, pickled_dataset_relative_path, dataset_size, validation_split=0.2,
                  batch_size=32, epochs=10, plot=True, save=True):
    dirname = os.path.dirname(__file__)

    (dataset_train, dataset_validate) = read_pickled_dataset(pickled_dataset_relative_path,
                                                             dataset_size,
                                                             validation_split=validation_split,
                                                             batch_size=batch_size)

    reconstructed_model = tf.keras.models.load_model(os.path.join(dirname, pickled_model_relative_path))

    history = reconstructed_model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_validate
    )

    if plot:
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                dirname,
                f"../../data/plots/{pickled_model_relative_path.split('/')[-1]}_{pickled_dataset_relative_path.split('/')[-1]}_loss_epochs{epochs}.png"))
        plt.close()

        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(
                dirname,
                f"../../data/plots/{pickled_model_relative_path.split('/')[-1]}_{pickled_dataset_relative_path.split('/')[-1]}_accuracy_epochs{epochs}.png"))
        plt.close()

    if save:
        init_model_save_path = os.path.join(dirname,
                                            f"retrained_{pickled_model_relative_path.split('/')[-1]}_{pickled_dataset_relative_path.split('/')[-1]}_epochs{epochs}.model")
        if not os.path.exists(init_model_save_path):
            tf.keras.models.save_model(reconstructed_model, init_model_save_path)
        else:
            while os.path.exists(init_model_save_path):
                init_model_save_path += ".redo"
            tf.keras.models.save_model(reconstructed_model, init_model_save_path)


def retrain_all_models():
    """
    this could work with different suffixes added per dataset
    .aug_mnist_e10 (e for epochs)
    .rl_mnist_e10
    .rl_complex_e10
    filenames could become unbearable at some point so this should be considered

    timestamp in suffix would make this superlong.. and xomplicated to read. but very specific
    test models not older than 24h for example

    this could be done in original retrain function, no changing of the original model name, just adding suffixes
    hmmm....

    :return:
    """
    pass
