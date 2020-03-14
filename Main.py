# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np

import gym

from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from keras.optimizers import Adam
import keras.callbacks
import numba as nb
from tensorboardX import SummaryWriter

from actor import Actor
from critic import Critic

ENV = 'gym_rosetta:protein-fold-v0'
CONTINUOUS = True

EPISODES = 10000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 8
NOISE = 0.15 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 1024
BATCH_SIZE = 128
NUM_ACTIONS = 64
NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 4e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_RESIDUE_ACTION, DUMMY_TORSION_ACTION, DUMMY_ANGLE_ACTION, DUMMY_VALUE = np.zeros((64,)),  np.zeros((3,)), np.zeros((1,)), np.zeros((1, 1))
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new

class Agent:
    def __init__(self):

        self.env = gym.make(ENV)
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        env_dim = self.get_state_size()
        action_space = self.env.action_space
        self.actor = Actor(env_dim, action_space, LR, NOISE, LOSS_CLIPPING)
        self.critic = Critic(env_dim, action_space, LR)

        self.episode = 0
        obs, reward, done, a = self.env.reset()
        self.observation = [obs[key] for key in obs.keys()]

        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.name = self.get_name()
        self.writer = SummaryWriter(self.name)
        self.gradient_steps = 0

    def get_state_size(self):
        shapes = {}
        for obs in self.env.observation_space.spaces.keys():
            shapes[obs] = self.env.observation_space[obs].shape
        return shapes

    def get_name(self):
        name = 'AllRuns/'
        if CONTINUOUS is True:
            name += 'continous/'
        else:
            name += 'discrete/'
        name += ENV
        return name

    def reset_env(self):
        self.episode += 1
        if self.episode % 20 == 0:
            self.val = True
            self.env.set_validation_mode(True)
        else:
            self.val = False
            self.env.set_validation_mode(False)

        obs, reward, done, a = self.env.reset()
        self.observation = [obs[key] for key in obs.keys()]

        self.reward = []

    def get_value(self):
        a, b, c, residue_mask = self.observation
        return self.critic.predict([[a], [b], [c]])

    def get_action_continuous(self):

        a, b, c, residue_mask = self.observation
        p = self.actor.predict_residue([[a], [b], [c], [residue_mask], DUMMY_VALUE, [DUMMY_RESIDUE_ACTION]])
        if self.val is False:
            residue_action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            residue_action = np.argmax(p[0])
        zeros = np.zeros(NUM_ACTIONS)
        zeros[residue_action] = 1

        torsion_pred = self.actor.predict_torsion([[a], [b], [c], [zeros], DUMMY_VALUE, [DUMMY_TORSION_ACTION]])
        if self.val is False:
            torsion_action = np.random.choice(3, p=np.nan_to_num(torsion_pred[0]))
        else:
            torsion_action = np.argmax(torsion_pred[0])
        torsion_log = np.zeros(3)
        torsion_log[torsion_action] = 1

        angle_pred = self.actor.predict_angle([[a], [b], [c], [zeros], [torsion_log], DUMMY_VALUE, [DUMMY_ANGLE_ACTION]])
        if self.val is False:
            angle_action = angle_pred[0] + np.random.normal(loc=0, scale=NOISE, size=angle_pred[0].shape)
        else:
            angle_action = angle_pred[0]
        action_matrix = [zeros, torsion_log, angle_action]
        return {"residue": residue_action, "torsion": torsion_action, "angles": angle_action}, action_matrix, [p[0], torsion_pred[0], angle_pred[0]]

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)

    def transform_output(self, batched_output):
        angle = []
        torsion = []
        residue = []
        for y in batched_output:
            angle.append(y[2])
            torsion.append(y[1])
            residue.append(y[0])
        return residue, torsion, angle

    def get_batch(self):
        batch = [[], [], []]
        input_batch = [[] for _ in range(4)]

        tmp_batch = [[], []]
        temp_input_batch = [[] for _ in range(4)]
        temp_value = []
        temp_reward = []
        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action_continuous()
            print(action)
            values = self.get_value()
            temp_value.append(values[0][0])

            obs_local, reward, done, info = self.env.step(action)
            observation = [obs_local[key] for key in obs_local.keys()]
            self.reward.append(reward)
            temp_reward.append(reward)
            for i, ob in enumerate(self.observation):
                temp_input_batch[i].append(ob)

            tmp_batch[0].append(action_matrix)
            tmp_batch[1].append(predicted_action)
            self.observation = observation

            if len(tmp_batch[0]) == 32:
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        action, pred = tmp_batch[0][i], tmp_batch[1][i]
                        batch[0].append(action)
                        batch[1].append(pred)
                        # batch[2].append(r)
                    for i in range(len(temp_input_batch)):
                        input_batch[i] = input_batch[i] + temp_input_batch[i]
                    lastgaelam = 0
                    mb_advs = np.zeros_like(temp_reward)
                    for t in reversed(range(len(tmp_batch[0]))):
                        if t == len(tmp_batch[0]) - 1:
                            nextnonterminal = 1 - done
                            nextvalues = values
                        else:
                            nextnonterminal = 1
                            nextvalues = temp_value[t + 1]
                        delta = temp_reward[t] + GAMMA * nextvalues * nextnonterminal - temp_value[t]
                        mb_advs[t] = lastgaelam = delta + GAMMA * nextnonterminal * lastgaelam
                    mb_returns = mb_advs + temp_value
                    batch[2] = np.concatenate((batch[2], mb_returns))
                tmp_batch = [[], []]
                temp_input_batch = [[] for _ in range(4)]
                temp_value = []
                temp_reward = []
                if done:
                    self.transform_reward()
                    self.reset_env()
        action, pred, returns = np.array(batch[0]), np.array(batch[1]), np.reshape(np.array(batch[2]), (len(batch[2]), 1))

        obs = input_batch
        return obs, action, pred, returns

    def run(self):
        valid_step = 0
        # tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True,
        #                                          write_images=True)
        while self.episode < EPISODES:
            obs, action, pred, returns = self.get_batch()
            action, pred, returns = action[:BUFFER_SIZE], pred[:BUFFER_SIZE], returns[:BUFFER_SIZE]
            for i in range(len(obs)):
                obs[i] = obs[i][:BUFFER_SIZE]

            residue_action, torsion_angle, angle_action = self.transform_output(action)
            pred_residue_action, pred_torsion_angle, pred_angle_action = self.transform_output(pred)

            values = self.critic.predict(obs[:3])
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            actor_loss_residue = self.actor.train_residue([*obs, advs, pred_residue_action], [residue_action],
                                          batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True)
            self.actor.train_torsion([*obs[:3], residue_action, advs, pred_torsion_angle], [torsion_angle],
                                                          batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                                                          verbose=True)
            actor_loss_angle = self.actor.train_angle([*obs[:3], residue_action, torsion_angle, advs, pred_angle_action],
                                                  [angle_action],
                                                  batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True)
            critic_loss = self.critic.train([*obs[:3]], [returns], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True)
            self.writer.add_scalar('Actor residue loss', actor_loss_residue.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Actor angle loss', actor_loss_angle.history['loss'][-1], self.gradient_steps)

            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
            if self.val:
                valid_step += 1
                self.actor.save('')

            self.gradient_steps += 1


if __name__ == '__main__':
    ag = Agent()
    ag.run()
