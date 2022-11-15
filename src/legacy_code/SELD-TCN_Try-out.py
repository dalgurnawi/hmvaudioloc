# # Importing standard libraries
# import matplotlib.pyplot as plt
# from Ipython.display import Audio, display
# import tensorflow as tf
# import tensorflow_io as tfio
# import numpy as np
# import librosa
#
# # Converting audio sample to numpy
#
# for sample in dataset_train_original.shuffle(2500).take(1):
#     audio = sample["audio"].numpy().astype("float32")
#     label = sample["label"].numpy()
#
#     #Convert to Mel Spectogram
#
#     mel = librosa.feature.melspectrogram(
#         y=audio,
#         sr=8000,
#         n_mels=64,
#         fmax=2000
#     )
#
# #Ensure data are the same length
#
# def proprocess(sample):
#     audio = sample["audio"]
#     label = sample["label"]
#
#     audio = tf.cast(audio, tf.float32)/32768
#
#     #Turn into spectrogram first
#
#     spectrogram = tfio.audio.spectrogram(
#         audio,
#         nfft=1024,
#         window=1024,
#         stride=64
#     )
#
#     # Turn into melspectrogram
#     spectrogram = tfio.audio.melscale(spectrogram,
#                                       rate=8000,
#                                       mels=64,
#                                       fmin=0,
#                                       fmax=2000
#                                       )#Applies fourier transformation
#
#     spectrogram /= tf.math.reduce_max(spectrogram) #picks the maximum and normalises the data
#
#     spectrogram = tf.expand_dims(spectrogram, axis =  -1) # increases dimensions from 2d to 3d
#
#     spectrogram = tf.image.resize(spectrogram, (64,64))
#
#     spectrogram = tf.transpose(spectrogram, perm(1,0,2)) #rearrange film
#
#     spectrogram = spectrogram[::-1,:,:]
#
#     return spectrogram, label
#
# dataset = dataset_train_original.map(lambda sample: preprocess(sample))
#
# for x,y in dataset.take(1):
#     x=x.numpy()
#     print(x.shape)
#     plt.imshow(x.squeeze(), cmap="inferno")
#
# dataset_train = dataset_train_original.map(lambda sample: preprocess(sample))
# dataset_train = dataset_train.cache() #no impact on neural network but speeds up processing time
# dataset_train = dataset_train.shuffle(2500)
# dataset_train = dataset_train.batch(25)
#
# dataset_validate = dataset_validate_original.map(lambda sample: preprocess(sample))
# dataset_validate = dataset_validate.cache()
# dataset_validate = dataset_validate.batch(25)
#
# epochs = 50
#
# from tensorflow.keras import models, layers
# from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input
# from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
# from keras.layers.recurrent import GRU
# from keras.layers.normalization import BatchNormalization
# from keras.models import Model
# from keras.layers.wrappers import TimeDistributed
# from keras.optimizers import Adam
# import keras
# keras.backend.set_image_data_format('channels_first')
# from IPython import embed
#
# # Start of model
# poolsize = (8, 8, 2) # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
# dropout_rate = 0.5
# model_input = (64,64,2) # 3D dimension set to 2 to represent stereo input
# kernel_size_res_block = (3,3,2)
# padding_res_block = 'same'
#
# sequence_length = 512,  # Feature sequence length
# batch_size = 16,  # Batch size
# nb_cnn2d_filt = 64,  # Number of CNN nodes, constant for each layer
# rnn_size = [128, 128],  # RNN contents, length of list = number of layers, list value = number of nodes
# fnn_size = [128],  # FNN contents, length of list = number of layers, list value = number of nodes
# loss_weights = [1., 50.],  # [sed, doa] weight for scaling the DNN outputs
# xyz_def_zero = True,  # Use default DOA Cartesian value x,y,z = 0,0,0
# nb_epochs = 500,  # Train for maximum epochs
#
# # Residual Block Layer
#
# # Applying the paper's 256 filters with a filters of size 3
#
# model = models.Sequential()
#
# def get_seldtcn_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size, fnn_size)
#
#
#     # CONVOLUTIONAL LAYERS =================================================================
#
#     for i, convCnt in enumerate(poolsize):
#         spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernal_size=(3,3), padding='same', input=model_input)
#         spec_cnn = BatchNormalization()(model_input)
#         spec_cnn = Activation('relu')(model_input)
#         spec_cnn = MaxPooling2D(poolsize)(model_input)
#         spec_cnn = Dropout(dropout_rate)(model_input)
#     spec_cnn = Permute((2,1,3))(model_input)
#     resblock_input = Reshape((data_in[-2],-1))(spec_cnn)
#
#
# # TCN layer ==============================================================================
#
# # residual blocks ------------------------------------------------------------------------
#
# skip_connections = []
#
# for d in range(10):
#     # 2D convolution
#     spec_conv2d = model.add(layers.Conv2D(256, kernel_size=kernel_size_res_block, padding= padding_res_block,
#                                           input_shape=(64, 64, 2))) # Dilation rate is missing
#     # Batch Normalisation
#     spec_conv2d = BatchNormalization()(spec_conv2d)
#
#     # Adding Activations
#
#     tanh_out = model.add(layers.Conv2D(256, kernel_size=kernel_size_res_block,
#                                        padding=padding_res_block, input=resblock_input, activation='tanh'))
#     sigm_out = model.add(layers.Conv2D(256, kernel_size=kernel_size_res_block,
#                                        padding=padding_res_block, input=resblock_input, activation='sigmoid'))
#
#     act_mult = model.add(layers.Multiply()([tanh_out,sigm_out]))
#
#     # Spatial dropout
#
#     spec_dropout = model.add(layers.Dropout(dropout_rate))(act_mult)
#
#     # 2D Convolution
#     skip_output = model.add(layers.Convolution2D(filters=128, kernel_size=kernel_size_res_block,
#                                                  padding=padding_res_block, input=spec_dropout))
#
#     res_output = model.add(layers)([resblock_input,skip_output])
#
#     if skip_output is not None:
#         skip_connections.append(skip_output)
#
#     resblock_input = res_output
#
#     # -----------------------------------------------------------------------------------
#
#     #Residual blocks sum
#     spec_sum = model.add(layers)(skip_connections)
#     spec_sum = model.add(layers(activation='relu', input=spec_sum))
#
#     #2D Convolutions
#     spec_conv2d_2 = model.add(layers.Conv2D(filters=128, kernel_size=(1,1), padding='same',
#                                             activation = 'relu', input=spec_sum))
#     spec_tcn = model.add(layers.Conv2D(filters=128, kernal_size=(1,1), padding='same',
#                                             activation = 'tanh', input=spec_conv2d_2))
#
#     # SED--------------------------------------------------------------------------------
#
#     # SED ==================================================================
#     sed = spec_tcn
#     for nb_fnn_filt in fnn_size:
#         sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
#     sed = Dropout(dropout_rate)(sed)
#     sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
#     sed = Activation('sigmoid', name='sed_out')(sed)
#
#     # DOA ==================================================================
#     doa = spec_tcn
#     for nb_fnn_filt in fnn_size:
#         doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
#     doa = Dropout(dropout_rate)(doa)
#
#     doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
#     doa = Activation('tanh', name='doa_out')(doa)
#
#     model = Model(inputs=spec_start, outputs=[sed, doa])
#     model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
#
#     model.summary()
#     return model
#
#
