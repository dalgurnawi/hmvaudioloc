# Importing standard libraries
import matplotlib.pyplot as plt
from Ipython.display import Audio, display
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import librosa

# Converting audio sample to numpy

for sample in dataset_train_original.shuffle(2500).take(1):
    audio = sample["audio"].numpy().astype("float32")
    label = sample["label"].numpy()

    #Convert to Mel Spectogram

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=8000,
        n_mels=64,
        fmax=2000
    )

#Ensure data are the same length

def proprocess(sample):
    audio = sample["audio"]
    label = sample["label"]

    audio = tf.cast(audio, tf.float32)/32768

    #Turn into spectrogram first

    spectrogram = tfio.audio.spectrogram(
        audio,
        nfft=1024,
        window=1024,
        stride=64
    )

    # Turn into melspectrogram
    spectrogram = tfio.audio.melscale(spectrogram,
                                      rate=8000,
                                      mels=64,
                                      fmin=0,
                                      fmax=2000
                                      )#Applies fourier transformation

    spectrogram /= tf.math.reduce_max(spectrogram) #picks the maximum and normalises the data

    spectrogram = tf.expand_dims(spectrogram, axis =  -1) # increases dimensions from 2d to 3d

    spectrogram = tf.image.resize(spectrogram, (64,64))

    spectrogram = tf.transpose(spectrogram, perm(1,0,2)) #rearrange film

    spectrogram = spectrogram[::-1,:,:]

    return spectrogram, label

dataset = dataset_train_original.map(lambda sample: preprocess(sample))

for x,y in dataset.take(1):
    x=x.numpy()
    print(x.shape)
    plt.imshow(x.squeeze(), cmap="inferno")

dataset_train = dataset_train_original.map(lambda sample: preprocess(sample))
dataset_train = dataset_train.cache() #no impact on neural network but speeds up processing time
dataset_train = dataset_train.shuffle(2500)
dataset_train = dataset_train.batch(25)

dataset_validate = dataset_validate_original.map(lambda sample: preprocess(sample))
dataset_validate = dataset_validate.cache()
dataset_validate = dataset_validate.batch(25)

epochs = 50

from tensorflow.keras import models, layers

model = models.Sequential()

model.add(layers.Conv2D(4, (3,3), activation="relu", padding= "same", input_shape=(64, 64, 2)))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(16, (3,3), activation="relu", padding="same"))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation="sigmoid"))
model.add(layers.Dense(8, activation="softmax"))
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_validate
)
