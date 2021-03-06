import sys

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout1D, add, concatenate, Softmax, Add, Activation, Dot, Multiply
from keras.layers import LSTM, Bidirectional, multiply, GlobalAveragePooling1D,RepeatVector
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model

ENTROPY_LOSS = 1e-4

def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return K.constant(out)
    return _initializer

def entropy(logits):
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

def prob_continous(y_true, y_pred, noise):
    var = tf.exp(noise)

    return 0.5 * tf.reduce_sum(tf.square((y_true - y_pred) / var), axis=-1) \
    + 0.5 * np.log(2.0 * np.pi) * y_true + noise

def proximal_policy_optimization_loss_continuous(advantage, old_prediction, noise, loss_clipping):
    def loss(y_true, y_pred):
        pred_mean, pred_logstd = tf.split(axis=1, num_or_size_splits=2, value=y_pred)
        old_mean, old_logstd = tf.split(axis=1, num_or_size_splits=2, value=old_prediction)

        var = K.square(pred_logstd)
        old_var = K.square(old_logstd)

        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        old_denom = K.sqrt(2 * pi * old_var)

        prob_num = K.exp(- K.square(y_true - pred_mean) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_mean) / (2 * old_var))

        prob = prob_num / denom
        old_prob = old_prob_num / old_denom
        r = prob / (old_prob + 1e-10)
        entropy = (pred_logstd + .5 * np.log(2.0 * np.pi * np.e)) * ENTROPY_LOSS

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage) + entropy)
    return loss

def proximal_policy_optimization_loss(advantage, old_prediction, loss_clipping):
    def loss(y_true, y_pred):
        # prob = categorical_crossentropy(y_true, y_pred)
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        entropy = -K.sum(y_pred * K.log(y_pred + 1e-10), axis=-1) * 0
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage) + ENTROPY_LOSS *entropy)
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
        self.model_residue, self.model_torsion, self.model_angle = self.network()

    def network(self):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        inp = Input(shape=self.env_backbone_dim)
        inp_acid = Input(shape=(self.env_amino_acid_dim))
        inp_residue = Input(shape=(self.act_residue_dim,))
        inp_torsion = Input(shape=(3,))
        inp_residue_mask = Input(shape=(self.act_residue_dim,))


        auxiliary_input = Input(shape=self.env_energy_dim)
        advantage = Input(shape=(1,))
        old_angle_prediction = Input(shape=(2,))
        old_torsion_prediction = Input(shape=(3,))
        old_residue_prediction = Input(shape=(self.act_residue_dim,))

        rrn = Bidirectional(LSTM(64, return_sequences=False))(inp)
        rrn_acid = Bidirectional(LSTM(256, return_sequences=False))(inp_acid)
        hidden = concatenate([rrn, rrn_acid])
        x = Dense(256, activation='relu')(hidden)
        x = concatenate([x, auxiliary_input])
        x = Dense(128, activation='relu')(x)

        # rrn = Bidirectional(LSTM(32, return_sequences=False))(hidden)
        hidden_torsion_model = concatenate([x, inp_residue])
        torsion_output_log = Dense(128, activation='relu')(hidden_torsion_model)

        torsion_output_softmax = Dense(3, activation='softmax', kernel_initializer=normc_initializer(0.01))(torsion_output_log)
        model_torsion = Model([inp, inp_acid, auxiliary_input, inp_residue, advantage, old_torsion_prediction], outputs=[torsion_output_softmax])
        model_torsion.compile(optimizer=Adam(lr=self.lr),
                              loss=[proximal_policy_optimization_loss(advantage, old_torsion_prediction,
                                                                      self.loss_clipping)])

        residue_selected = Dense(self.act_residue_dim, activation='relu', kernel_initializer=normc_initializer(0.1))(x)
        residue_selected = Add()([residue_selected, inp_residue_mask])
        residue_selected = Multiply()([residue_selected, inp_residue_mask])

        def custom_activation(x):
            x, mask = x
            x = K.switch(tf.is_nan(x), K.zeros_like(x), x)  # prevent nan values
            x = K.switch(K.equal(mask, 0), K.zeros_like(x), K.exp(x))
            return x / K.sum(x, axis=-1, keepdims=True)
        residue_selected_softmax = Activation(custom_activation)([residue_selected, inp_residue_mask])
        # residue_selected_softmax = Softmax(input_mask=[3])(residue_selected)

        torsion_output = Dense(64, activation='relu')(hidden_torsion_model)
        hidden_angle_model = concatenate([torsion_output, inp_torsion])

        hidden_angle_model = Dense(64, activation='relu')(hidden_angle_model)

        angle_mover_mean = Dense(1, activation='tanh', kernel_initializer=normc_initializer(0.01))(hidden_angle_model)
        angle_mover_std = Dense(1, activation='sigmoid', kernel_initializer=normc_initializer(0.1))(hidden_angle_model)
        angle_mover = concatenate([angle_mover_mean, angle_mover_std])
        model_residue = Model([inp, inp_acid, auxiliary_input, inp_residue_mask, advantage, old_residue_prediction], outputs=[residue_selected_softmax])

        model_angle = Model([inp, inp_acid, auxiliary_input, inp_residue, inp_torsion, advantage, old_angle_prediction], outputs=[angle_mover])

        model_residue.compile(optimizer=Adam(lr=self.lr),
                      loss=proximal_policy_optimization_loss(advantage, old_residue_prediction, self.loss_clipping))
        model_residue.summary()
        model_angle.summary()

        model_angle.compile(optimizer=Adam(lr=self.lr),
                              loss=[proximal_policy_optimization_loss_continuous(
                                        advantage, old_angle_prediction, self.noise, self.loss_clipping)])
        plot_model(model_angle, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model_residue, model_torsion, model_angle

    def predict_residue(self, state):
        """ Action prediction
        """
        return self.model_residue.predict(state)

    def train_residue(self, input, out, batch_size=64, shuffle=True, epochs=8, verbose=False,  callbacks=None):
        """ Actor Training
        """
        return self.model_residue.fit(input, out, batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=verbose,  callbacks=callbacks)

    def predict_torsion(self, state):
        """ Action prediction
        """
        return self.model_torsion.predict(state)

    def train_torsion(self, input, out, batch_size=64, shuffle=True, epochs=8, verbose=False, callbacks=None):
        """ Actor Training
        """
        return self.model_torsion.fit(input, out, batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=verbose,
                                    callbacks=callbacks)

    def predict_angle(self, state):
        """ Action prediction
        """
        return self.model_angle.predict(state)

    def train_angle(self, input, out, batch_size=64, shuffle=True, epochs=8, verbose=False,  callbacks=None):
        """ Actor Training
        """
        return self.model_angle.fit(input, out, batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=verbose,  callbacks=callbacks)



    def save(self, path):
        self.model_angle.save_weights(path + '_actor.h5')
        self.model_residue.save_weights(path + '_actor-residue.h5')


    def load_weights(self, path_residue, path_angle):
        self.model_angle.load_weights(path_residue)
        self.model_residue.load_weights(path_angle)

