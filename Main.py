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

EPISODES = 100000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 0.5 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 64
BATCH_SIZE = 32
NUM_ACTIONS = 128
NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_RESIDUE_ACTION, DUMMY_ANGLE_ACTION, DUMMY_VALUE = np.zeros((128,)),  np.zeros((3,)), np.zeros((1, 1))


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
        else:
            self.val = False
        obs, reward, done, a = self.env.reset()
        self.observation = [obs[key] for key in obs.keys()]

        self.reward = []

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_RESIDUE_ACTION, DUMMY_ANGLE_ACTION])
        if self.val is False:

            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_action_continuous(self):

        a, b, c = self.observation
        p = self.actor.predict([[a], [b], [c], DUMMY_VALUE, [DUMMY_RESIDUE_ACTION], [DUMMY_ANGLE_ACTION]])
        if self.val is False:
            angle_action = p[1][0] + np.random.normal(loc=0, scale=NOISE, size=p[1][0].shape)
            residue_action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0][0]))
        else:
            angle_action = p[1][0]
            residue_action = np.argmax(p[0][0])
        print(residue_action)

        zeros = np.zeros(NUM_ACTIONS)
        zeros[residue_action] = 1
        action_matrix = [zeros, angle_action]
        return {"residue": residue_action, "angles": angle_action}, action_matrix, [p[0][0], p[1][0]]

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def transform_output(self, batched_output):
        angle = []
        residue = []
        for y in batched_output:
            angle.append(y[1])
            residue.append(y[0])
        return residue, angle

    def get_batch(self):
        batch = [[], [], []]
        input_batch = [[] for _ in range(3)]

        tmp_batch = [[], []]
        temp_input_batch = [[] for _ in range(3)]

        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action_continuous()
            obs_local, reward, done, info = self.env.step(action)
            observation = [obs_local[key] for key in obs_local.keys()]
            print(reward)
            self.reward.append(reward)
            for i, ob in enumerate(self.observation):
                temp_input_batch[i].append(ob)

            tmp_batch[0].append(action_matrix)
            tmp_batch[1].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        action, pred = tmp_batch[0][i], tmp_batch[1][i]
                        r = self.reward[i]
                        batch[0].append(action)
                        batch[1].append(pred)
                        batch[2].append(r)
                    for i in range(len(temp_input_batch)):
                        input_batch[i] = input_batch[i] + temp_input_batch[i]
                tmp_batch = [[], []]
                temp_input_batch = [[] for _ in range(3)]
                self.reset_env()
        action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.reshape(np.array(batch[2]), (len(batch[2]), 1))

        obs = input_batch
        return obs, action, pred, reward

    def run(self):
        valid_step = 0
        # tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True,
        #                                          write_images=True)
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            action, pred, reward = action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            for i in range(len(obs)):
                obs[i] = obs[i][:BUFFER_SIZE]
            old_prediction = pred
            residue_action, angle_action = self.transform_output(action)
            pred_residue_action, pred_angle_action = self.transform_output(pred)

            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            actor_loss = self.actor.train([*obs, advantage, pred_residue_action, pred_angle_action], [residue_action, angle_action],
                                          batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True)
            critic_loss = self.critic.train([*obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Reward history', np.mean(reward), self.gradient_steps)
            if self.val:
                valid_step += 1
                self.writer.add_scalar('validation Reward history', np.mean(reward), valid_step)

            self.gradient_steps += 1


if __name__ == '__main__':
    ag = Agent()
    ag.run()
