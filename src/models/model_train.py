import tensorflow as tf
import matplotlib.pyplot as plt
import os

from src.models.model_architectures.model_baseline_vanilla import create_baseline_vanilla_model
from src.models.model_architectures.model_baseline import create_baseline_model
from src.models.model_architectures.model_baseline_gaussian_noise import create_baseline_gaussian_noise_model

from src.models.model_architectures.model_paper_code_1 import create_paper_code_1_model
from src.models.model_architectures.model_paper_code_2 import create_paper_code_2_model
from src.models.model_architectures.model_paper_code_3 import create_paper_code_3_model
from src.models.model_architectures.model_paper_code_4 import create_paper_code_4_model
from src.models.model_architectures.model_paper_code_5 import create_paper_code_5_model
from src.models.model_architectures.model_paper_code_6 import create_paper_code_6_model
from src.models.model_architectures.model_paper_code_7 import create_paper_code_7_model
from src.models.model_architectures.model_paper_code_8 import create_paper_code_8_model
from src.models.model_architectures.model_paper_code_9 import create_paper_code_9_model
from src.models.model_architectures.model_paper_code_10 import create_paper_code_10_model
from src.models.model_architectures.model_paper_code_11 import create_paper_code_11_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compile_model(model_choice="baseline_model", optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"):

    if model_choice == "baseline_model":
        model = create_baseline_model()
    elif model_choice == "baseline_model_vanilla":
        model = create_baseline_vanilla_model()
    elif model_choice == "baseline_model_gaussian_noise":
        model = create_baseline_gaussian_noise_model()

    elif model_choice == "model_paper_code_1":
        model = create_paper_code_1_model()
    elif model_choice == "model_paper_code_2":
        model = create_paper_code_2_model()
    elif model_choice == "model_paper_code_3":
        model = create_paper_code_3_model()
    elif model_choice == "model_paper_code_4":
        model = create_paper_code_4_model()
    elif model_choice == "model_paper_code_5":
        model = create_paper_code_5_model()
    elif model_choice == "model_paper_code_6":
        model = create_paper_code_6_model()
    elif model_choice == "model_paper_code_7":
        model = create_paper_code_7_model()
    elif model_choice == "model_paper_code_8":
        model = create_paper_code_8_model()
    elif model_choice == "model_paper_code_9":
        model = create_paper_code_9_model()
    elif model_choice == "model_paper_code_10":
        model = create_paper_code_10_model()
    elif model_choice == "model_paper_code_11":
        model = create_paper_code_11_model()

    # elif model_choice == "yamnet-crepe":
    #     mdoel = create_yamnet_crepe_model()

    else:
        raise Exception("Model chosen does not exist in available model architectures")

    model.summary()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metrics]
    )

    return model


def train_model_plot_and_save(dataset_train, dataset_validate, model_choice="baseline_model", optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy", epochs=10, plot=True, save=True):

    model = compile_model(model_choice=model_choice,
                          optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_validate
    )

    dirname = os.path.dirname(__file__)

    if plot:
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                dirname,
                f"../../data/plots/{model_choice}_loss_epochs{epochs}.png"))
        plt.close()

        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.legend()
        plt.savefig(
            os.path.join(
                dirname,
                f"../../data/plots/{model_choice}_accuracy_epochs{epochs}.png"))
        plt.close()

    if save:
        tf.keras.models.save_model(model, os.path.join(dirname, f"../../data/pickled_models/{model_choice}_epochs{epochs}.model"))
