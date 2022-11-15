import matplotlib as plt
import os


dirname = os.path.dirname(__file__)


def plot_single_training(history, model_choice, epochs):
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


def plot_all_training():
    pass
