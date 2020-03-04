import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout1D, add, concatenate, Softmax
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,RepeatVector
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

ENTROPY_LOSS = 5e-3


def proximal_policy_optimization_loss_continuous(advantage, old_prediction, noise, loss_clipping):
    def loss(y_true, y_pred):
        var = K.square(noise)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))

        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = K.mean(prob/(old_prob + 1e-10))

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage))
    return loss

def proximal_policy_optimization_loss(advantage, old_prediction, loss_clipping):
    def loss(y_true, y_pred):
        prob = categorical_crossentropy(y_true, y_pred)
        # prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = categorical_crossentropy(y_true, old_prediction)
        r = K.exp(old_prob - prob)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss

class Actor:
    """ Actor Network for the DDPG Algorithm
    """

    def __init__(self, inp_dim, out_dim, lr, noise, loss_clipping):
        self.env_backbone_dim = inp_dim["backbone"]
        self.env_amino_acid_dim = inp_dim["amino_acids"]
        self.env_energy_dim = inp_dim["energy"]
        self.act_residue_dim = out_dim["residue"].n
        self.act_angle_dim = out_dim["angles"].shape

        self.lr = lr
        self.noise = noise
        self.loss_clipping = loss_clipping
        self.model = self.network()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=self.env_backbone_dim)
        inp_acid = Input(shape=self.env_amino_acid_dim)

        auxiliary_input = Input(shape=self.env_energy_dim)
        advantage = Input(shape=(1,))
        old_angle_prediction = Input(shape=self.act_angle_dim)
        old_residue_prediction = Input(shape=(self.act_residue_dim,))

        rrn = Bidirectional(LSTM(64, return_sequences=False))(inp)
        rrn_acid = Bidirectional(LSTM(64, return_sequences=False))(inp_acid)
        hidden = concatenate([rrn, rrn_acid], axis=1)

        # rrn = Bidirectional(LSTM(32, return_sequences=False))(hidden)

        # rnn = Dense(128, activation='relu')(rrn)
        x = Dense(128, activation='relu')(hidden)
        x = concatenate([x, auxiliary_input])
        x = Dense(64, activation='relu')(x)

        residue_selected = Dense(self.act_residue_dim, activation='softmax')(x)
        angle_mover = Dense(3, activation='tanh')(x)


        # x = GaussianNoise(1.0)(x)


        model = Model([inp, inp_acid, auxiliary_input, advantage, old_residue_prediction, old_angle_prediction], outputs=[residue_selected, angle_mover])

        model.compile(optimizer=Adam(lr=self.lr),
                      loss=[proximal_policy_optimization_loss(advantage, old_residue_prediction, self.loss_clipping),
                            proximal_policy_optimization_loss_continuous(
                          advantage, old_angle_prediction, self.noise, self.loss_clipping)])
        model.summary()

        return model

    def predict(self, state):
        """ Action prediction
        """
        return self.model.predict(state)


    def train(self, input, out, batch_size=64, shuffle=True, epochs=10, verbose=False,  callbacks=None):
        """ Actor Training
        """
        return self.model.fit(input, out, batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=verbose,  callbacks=callbacks)



    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
