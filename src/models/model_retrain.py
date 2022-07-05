import tensorflow as tf
import matplotlib.pyplot as plt

def reatrain_model(model_name, dataset_name, epochs=10):


    dataset = tf.data.experimental.load("pickled_real_world_dataset.dat")
    dataset = dataset.shuffle(24000)

    dataset_train = dataset.take(21600)
    dataset_validate = dataset.skip(21600)

    dataset_train = dataset_train.cache()
    dataset_train = dataset_train.shuffle(21600)
    dataset_train = dataset_train.batch(32)

    dataset_validate = dataset_validate.cache()
    dataset_validate = dataset_validate.batch(32)


    reconstructed_model = tf.keras.models.load_model('gaussian_noise_dropout.model')

    history = reconstructed_model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_validate
    )

    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.savefig("history1_retrain_dropout_Gnoise_real_world.png")
    plt.close()

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.savefig("history2_retrain_dropout_Gnoise_real_world.png")
    plt.close()

    tf.keras.models.save_model(reconstructed_model, 'real_world_retrained_gnoise_dropout.model')

