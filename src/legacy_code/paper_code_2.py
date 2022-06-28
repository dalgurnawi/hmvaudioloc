import tensorflow as tf
from keras.layers import Input
from keras.models import Model
import  matplotlib.pyplot as plt

dataset = tf.data.experimental.load("pickled_dataset.dat")
dataset = dataset.shuffle(72000)

dataset_train = dataset.take(57600)
dataset_validate = dataset.skip(57600)

dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(57600)
dataset_train = dataset_train.batch(32)

dataset_validate = dataset_validate.cache()
dataset_validate = dataset_validate.batch(32)



# SELD-TCN MODEL :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_seldtcn_model(dropout_rate):

    # model definition
    #spec_start = tf.keras.Sequential()
    spec_start = Input(shape=( 64, 64, 2))
    spec_cnn = spec_start
    
    # CONVOLUTIONAL LAYERS =========================================================
    #First convolutional layer
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Dropout(0)(spec_cnn)
    
    #Second convolutional layer
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 8), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Dropout(0)(spec_cnn)  
    
    #Third convolutional layer
    spec_cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(spec_cnn)
    spec_cnn = tf.keras.layers.BatchNormalization()(spec_cnn)
    spec_cnn = tf.keras.layers.Activation('relu')(spec_cnn)
    spec_cnn = tf.keras.layers.MaxPooling2D(pool_size=(1,2), padding="same")(spec_cnn)
    spec_cnn = tf.keras.layers.Dropout(0)(spec_cnn)  
    
    #resblock_input = Permute((2, 1, 3))(spec_cnn)
    #resblock_input = Reshape((64, -1))(spec_cnn)
    resblock_input = spec_cnn



    # TCN layer ===================================================================

    # residual blocks ------------------------
    skip_connections = []

    for d in range(8):

        # 2D convolution
        spec_conv2d = tf.keras.layers.Convolution2D(filters=256,
                                                kernel_size=(3,3),
                                                padding='same',
                                                dilation_rate=2**d)(resblock_input)
        spec_conv2d = tf.keras.layers.BatchNormalization()(spec_conv2d)

        # activations
        tanh_out = tf.keras.layers.Activation('tanh')(spec_conv2d)
        sigm_out = tf.keras.layers.Activation('sigmoid')(spec_conv2d)
        spec_act = tf.keras.layers.Multiply()([tanh_out, sigm_out])

        # spatial dropout
        spec_drop = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)(spec_act)

        # 2D convolution
        skip_output = tf.keras.layers.Convolution2D(filters=128,
                                                 kernel_size=(1,1),
                                                 padding='same')(spec_drop)

        res_output = tf.keras.layers.Add()([resblock_input, skip_output])

        if skip_output is not None:
            skip_connections.append(skip_output)

        resblock_input = res_output
    # ---------------------------------------

    # Residual blocks sum
    spec_sum = tf.keras.layers.Add()(skip_connections)
    spec_sum = tf.keras.layers.Activation('relu')(spec_sum)

    # 2D convolution
    spec_conv2d_2 = tf.keras.layers.Convolution2D(filters=128,
                                          kernel_size=(1,1),
                                          padding='same')(spec_sum)
    spec_conv2d_2 = tf.keras.layers.Activation('relu')(spec_conv2d_2)

    # 2D convolution
    spec_tcn = tf.keras.layers.Convolution2D(filters=128,
                                          kernel_size=(1,1),
                                          padding='same')(spec_conv2d_2)
    spec_tcn = tf.keras.layers.Activation('tanh')(spec_tcn)

    # Output ==================================================================
    #doa = spec_tcn
    #doa = tf.keras.layers.Activation('tanh')(spec_tcn)
    #doa = tf.keras.layers.Activation('softmax', name='doa_out')(doa)

    doa = tf.keras.layers.Flatten()(spec_tcn)
    #doa = tf.keras.layers.Dense(64, activation="sigmoid")(doa)
    doa = tf.keras.layers.Dense(8, activation="softmax")(doa)

    model = Model(inputs=spec_start, outputs=doa)
    model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics='accuracy')

    model.summary()
    return model


poolsize = (1,8) # CNN pooling, length of list = number of CNN layers, list value = pooling per layer. Changed from 2 to 1 to fit Tristan Data
dropout_rate = 0.5
model_input = (64, 64, 2) # 3D dimension set to 2 to represent stereo input # Changed to 1 to fit Tristan Data
kernel_size_res_block = (3,3,1) # Changed to 1 to fit Tristan Data
padding_res_block = 'same'
kernel_conv_block = (3,3,1)

sequence_length = 512,  # Feature sequence length
batch_size = 16,  # Batch size
nb_cnn2d_filt = 64,  # Number of CNN nodes, constant for each layer
rnn_size = [128, 128],  # RNN contents, length of list = number of layers, list value = number of nodes
fnn_size = [128],  # FNN contents, length of list = number of layers, list value = number of nodes
loss_weights = [1., 50.],  # [sed, doa] weight for scaling the DNN outputs
xyz_def_zero = True,  # Use default DOA Cartesian value x,y,z = 0,0,0
nb_epochs = 500,  # Train for maximum epochs


# sequence_length=512,        # Feature sequence length
#     batch_size=16,              # Batch size
#     dropout_rate=0.0,           # Dropout rate, constant for all layers
#     nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
#     pool_size=[8, 8, 2],        # CNN pooling, length of list = number of CNN layers, list value = pooling per layer
#     rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
#     fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
#     loss_weights=[1., 50.],     # [sed, doa] weight for scaling the DNN outputs
#     xyz_def_zero=True,          # Use default DOA Cartesian value x,y,z = 0,0,0
#     nb_epochs=500,              # Train for maximum epochs

model = get_seldtcn_model(dropout_rate=0.5)

epochs = 10
history = model.fit(dataset_train,
                    epochs=epochs,
                    validation_data=dataset_validate)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.savefig("history1_paper_code_2.png")
plt.close()

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.savefig("history2_paper_code_2.png")
plt.close()

model.save('paper_code_2.model')

