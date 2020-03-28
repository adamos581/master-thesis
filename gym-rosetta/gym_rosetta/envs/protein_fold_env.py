import json
import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from pyrosetta.teaching import *
from pyrosetta import init, pose_from_pdb, pose_from_sequence
from pyrosetta.toolbox import cleanATOM
from pyrosetta.rosetta.core.id import AtomID
from enum import Enum
from pyrosetta.rosetta.core.scoring import CA_rmsd
import numpy as np


init()
from pyrosetta.toolbox import cleanATOM
ANGLE_MOVE = [-90, -45, -10, -1, 1, 10, 45, 90]
import logging
logger = logging.getLogger(__name__)

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'empty'
]

MAX_LENGTH = 64

class ProteinFoldEnv(gym.Env, utils.EzPickle):

    def __init__(self, max_move_amount=1000):
        super(ProteinFoldEnv, self).__init__()
        self.reward = 0.0
        self.prev_ca_rmsd = None
        self.protein_pose = None
        self._configure_environment()
        self.is_finished = False
        self.move_counter = 0
        self.max_move_amount = max_move_amount
        self.name = None
        self.observation_space = spaces.Dict({
            "energy": spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=np.float32),
            "backbone": spaces.Box(low=-180, high=180, shape=(MAX_LENGTH, 3)),
            "amino_acids": spaces.Box(low=0, high=1, shape=(MAX_LENGTH, len(RESIDUE_LETTERS))),
            "residue_mask": spaces.Discrete(MAX_LENGTH)
        })
        self.action_space = spaces.Dict({
            "angles": spaces.Box(low=-1, high=1, shape=(3,)),
            "torsion": spaces.Discrete(3),
            "residue": spaces.Discrete(MAX_LENGTH)
        })
        self.encoded_residue_sequence = None
        self.scorefxn = get_fa_scorefxn()
        self.best_distance = np.inf
        self.validation = False
        self.best_conformations_dict = {}
        self.start_distance = 1000
        self.residue_mask = None
        self.shuffle = False
        self.achieved_goal_counter = 0


    def save_best_matches(self):
        with open('data.json', 'w') as fp:
            json.dump(self.best_conformations_dict, fp)

    def _configure_environment(self):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
        self._start_rossetta()

    def _start_rossetta(self):
        init()

    def _get_ca_metric(self, coordinates, target):
        return CA_rmsd(coordinates, target)

    def _move_on_torsion(self, residue_number, move_pose, get_angle, set_angle):
        new_torsion_position = get_angle(residue_number) + move_pose
        set_angle(residue_number, new_torsion_position)

    def _move(self, action):
        if action is not None:
            self.move_counter += 1
            residue_number = action["residue"] + 1
            torsion_number = action["torsion"]
            move_pose_index = action["angles"]
            move_pose = ANGLE_MOVE[move_pose_index]

            if residue_number < self.protein_pose.total_residue():
                if torsion_number == 0:
                    self._move_on_torsion(residue_number, move_pose, self.protein_pose.phi, self.protein_pose.set_phi)
                if torsion_number == 1:
                    self._move_on_torsion(residue_number, move_pose, self.protein_pose.omega, self.protein_pose.set_omega)
                if torsion_number == 2:
                    self._move_on_torsion(residue_number, move_pose, self.protein_pose.psi, self.protein_pose.set_psi)

                return 0.0
            else:
                return -0.5
        return 0.0
    def _encode_residues(self, sequence):
        encoded_residues = []
        for res in sequence:
            temp = np.zeros(len(RESIDUE_LETTERS))
            temp[RESIDUE_LETTERS.index(res)] = 1
            encoded_residues.append(temp)
        for zero in range(MAX_LENGTH - len(sequence)):
            temp = np.zeros(len(RESIDUE_LETTERS))
            temp[len(RESIDUE_LETTERS) - 1] = 1

            encoded_residues.append(temp)
        return encoded_residues

    def create_residue_mask(self, sequence):
        residues_mask = np.ones(len(sequence) - 2 ) * 1
        zero = np.zeros(MAX_LENGTH - len(sequence) + 1)
        return np.concatenate(([0], residues_mask, zero),)

    def write_best_conformation(self, distance):
        if self.name not in self.best_conformations_dict.keys():
            self.best_conformations_dict[self.name] = [distance]
        else:
            self.best_conformations_dict[self.name].append(distance)

    def step(self, action):
        penalty = self._move(action)
        ob = self._get_state()
        done = False
        reward = penalty

        distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)
        if self.prev_ca_rmsd:
            reward += self.prev_ca_rmsd - distance
        if self.best_distance > distance:
            # if distance < 3.0:
            #     reward += 1
            self.best_distance = distance
            print("best distance: {} ".format(distance))

        self.prev_ca_rmsd = distance
        if distance < self.start_distance * 0.1:
            self.achieved_goal_counter += 1
            reward += 5
            if self.achieved_goal_counter > 25:
                done = True

        if self.move_counter >= 1024:
            done = True

        return [ob, reward, done, {}]

    def reset(self):
        if self.start_distance > self.best_distance:
            self.write_best_conformation(self.best_distance)
        if self.validation:
            self.name = np.random.choice(os.listdir('protein_data/short_valid'))
            dir = 'protein_data/short_valid'
        else:
            self.name = np.random.choice(os.listdir('protein_data/short'))
            dir = 'protein_data/short'

        print('chosen protein: ' + self.name)
        self.target_protein_pose = pose_from_pdb(os.path.join(dir, self.name))
        if not self.shuffle:
            self.protein_pose = pose_from_sequence(self.target_protein_pose.sequence())
        # else :

        self.move_counter = 0
        self.reward = 0.0
        self.prev_ca_rmsd = None
        self.achieved_goal_counter = 0

        self.best_distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)
        self.start_distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)

        self.encoded_residue_sequence = self._encode_residues(self.target_protein_pose.sequence())
        self.residue_mask = self.create_residue_mask(self.target_protein_pose.sequence())
        self.save_best_matches()
        return self.step(None)

    def difference_energy(self):
        target_score = self.scorefxn(self.target_protein_pose)
        return (self.scorefxn(self.protein_pose) - target_score ) / target_score

    def _get_state(self):
        psis = np.divide([self.target_protein_pose.psi(i + 1) - self.protein_pose.psi(i + 1) for i in range(self.protein_pose.total_residue())], 180.0)
        omegas = np.divide([self.target_protein_pose.omega(i + 1) - self.protein_pose.omega(i + 1) for i in range(self.protein_pose.total_residue())], 180.0)
        phis = np.divide([self.target_protein_pose.phi(i + 1) - self.protein_pose.phi(i + 1) for i in range(self.protein_pose.total_residue())], 180)
        rest_zeros = MAX_LENGTH - len(self.target_protein_pose.sequence())
        psis = np.concatenate((psis, np.zeros(rest_zeros)))
        omegas = np.concatenate((omegas, np.zeros(rest_zeros)))
        phis = np.concatenate((phis, np.zeros(rest_zeros)))

        backbone_geometry = [list(a) for a in zip(psis, omegas, phis)]
        a = self.difference_energy()
        return {
            "backbone": backbone_geometry,
            "amino_acids": self.encoded_residue_sequence,
            "energy": self.difference_energy(),
            "residue_mask": self.residue_mask
        }

    def set_validation_mode(self, is_valid):
        self.validation = is_valid


NUM_ACTIONS = 3
END_GAME, ROTATE,TURN, = list(range(NUM_ACTIONS))