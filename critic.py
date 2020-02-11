import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr):
        # Dimensions and Hyperparams
        self.env_backbone_dim = inp_dim["backbone"]
        self.env_amino_acid_dim = inp_dim["amino_acids"]
        self.env_energy_dim= inp_dim["energy"]
        self.act_dim = out_dim
        self.lr = lr
        # Build models and target models
        self.model = self.network()
        self.model.compile(Adam(self.lr * 10), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        inp = Input(shape=self.env_backbone_dim)
        inp_acid = Input(shape=self.env_amino_acid_dim)
        auxiliary_input = Input(shape=self.env_energy_dim)

        rrn = Bidirectional(LSTM(64, return_sequences=True))(inp)
        rrn_acid = Bidirectional(LSTM(64, return_sequences=True))(inp_acid)
        hidden = concatenate([rrn, rrn_acid], axis=2)

        rrn = Bidirectional(LSTM(32, return_sequences=False))(hidden)

        rnn = Dense(128, activation='relu')(rrn)
        x = concatenate([rnn, auxiliary_input])
        x = Dense(32, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)

        return Model([inp, inp_acid, auxiliary_input], out)

    def predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.model.predict(inp)

    def train(self, input, out, batch_size=32, shuffle=True, epochs=10, verbose=False):
        """ Actor Training
        """
        return self.model.fit(input, out, batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=verbose)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)