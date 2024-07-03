import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model




class AutoEncoder:
    def __init__(self):
        os.environ['PYTHONHASHSEED'] = str(0)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        np.random.seed(12345)
        random.seed(12345)
        tf.random.set_seed(12345)

    def fit_autoencoder(self, data):
        input_dim = data.shape[2]
        hidden_dim = int(0.2 * input_dim) + 1

        input_layer = Input(shape=(input_dim,))
        hidden_layer = Dense(hidden_dim, activation='tanh')(input_layer)
        output_layer = Dense(input_dim, activation='linear')(hidden_layer)

        input_data = data.reshape(-1, input_dim)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        #print(autoencoder.summary())

        autoencoder.fit(input_data, input_data, epochs=10, batch_size=8)

        self.weights = autoencoder.get_weights()[0]
        self.input_data = input_data

    def encode_data(self, data):
        encoded_data = self.autoencoder.predict(data)
        return encoded_data

    def decode_data(self, data):
        decoded_data = self.autoencoder.predict(data)
        return decoded_data





