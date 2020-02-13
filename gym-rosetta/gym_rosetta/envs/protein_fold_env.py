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


import logging
logger = logging.getLogger(__name__)

RESIDUE_LETTERS = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'
]

MAX_LENGTH = 128

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
        self.observation_space = spaces.Dict({
            "energy": spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=np.float32),
            "backbone": spaces.Box(low=-180, high=180, shape=(MAX_LENGTH, 3)),
            "amino_acids": spaces.Box(low=0, high=1, shape=(MAX_LENGTH, len(RESIDUE_LETTERS)))
        })
        self.action_space = spaces.Dict({
            "angles": spaces.Box(low=-1, high=1, shape=(MAX_LENGTH, 3)),
            "residue": spaces.Discrete(MAX_LENGTH)
        })
        self.encoded_residue_sequence = None
        self.scorefxn = get_fa_scorefxn()
        # self.viewer = None
        # self.server_process = None
        # self.server_port = None
        # self.hfo_path = hfo_py.get_hfo_path()
        # self._configure_environment()
        # self.env = hfo_py.HFOEnvironment()
        # self.env.connectToServer(config_dir=hfo_py.get_config_path())
        # self.observation_space = spaces.Box(low=-1, high=1,
        #                                     shape=(self.env.getStateSize()))
        # # Action space omits the Tackle/Catch actions, which are useful on defense
        # self.action_space = spaces.Tuple((spaces.Discrete(3),
        #                                   spaces.Box(low=0, high=100, shape=1),
        #                                   spaces.Box(low=-180, high=180, shape=1),
        #                                   spaces.Box(low=-180, high=180, shape=1),
        #                                   spaces.Box(low=0, high=100, shape=1),
        #                                   spaces.Box(low=-180, high=180, shape=1)))
        # self.status = hfo_py.IN_GAME


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

    def _move(self, action):
        if action is not None:
            self.move_counter += 1
            residue_number = action["residue"]
            if residue_number < self.protein_pose.total_residue():
                residue_number += 1
                move_pose = action["angles"]

                phi_angle_value = move_pose[0] * 180.0
                omega_angle_value = move_pose[1] * 180.0
                psi_angle_value = move_pose[2] * 180.0
                self.protein_pose.set_phi(residue_number, phi_angle_value)
                self.protein_pose.set_omega(residue_number, omega_angle_value)
                self.protein_pose.set_psi(residue_number, psi_angle_value)

    def _encode_residues(self, sequence):
        encoded_residues = []
        for res in sequence:
            temp = np.zeros(len(RESIDUE_LETTERS))
            temp[RESIDUE_LETTERS.index(res)] = 1
            encoded_residues.append(temp)
        for zero in range(MAX_LENGTH - len(sequence)):
            encoded_residues.append(np.zeros(len(RESIDUE_LETTERS)))
        return encoded_residues

    def step(self, action):
        self._move(action)
        ob = self._get_state()
        done = False
        reward = 0.0

        distance = self._get_ca_metric(self.protein_pose, self.target_protein_pose)
        if self.prev_ca_rmsd:
            reward = self.prev_ca_rmsd - distance
        self.prev_ca_rmsd = distance
        if self.move_counter > 32:
            done = True

        return [ob, reward, done, {}]

    def reset(self):
        self.target_protein_pose = pose_from_pdb("2y29.pdb")
        self.protein_pose = pose_from_sequence(self.target_protein_pose.sequence())

        self.move_counter = 0
        self.reward = 0.0
        self.prev_ca_rmsd = None

        self.encoded_residue_sequence = self._encode_residues(self.target_protein_pose.sequence())

        return self.step(None)

    def _get_state(self):
        psis = np.divide([self.protein_pose.psi(i + 1) for i in range(self.protein_pose.total_residue())], 180.0)
        omegas = np.divide([self.protein_pose.omega(i + 1) for i in range(self.protein_pose.total_residue())], 180.0)
        phis = np.divide([self.protein_pose.phi(i + 1) for i in range(self.protein_pose.total_residue())], 180)
        rest_zeros = MAX_LENGTH - len(self.target_protein_pose.sequence())
        psis = np.concatenate((psis, np.zeros(rest_zeros)))
        omegas = np.concatenate((omegas, np.zeros(rest_zeros)))
        phis = np.concatenate((phis, np.zeros(rest_zeros)))

        backbone_geometry = [list(a) for a in zip(psis, omegas, phis)]
        return {
            "backbone": backbone_geometry,
            "amino_acids": self.encoded_residue_sequence,
            "energy": self.scorefxn(self.protein_pose)
        }


NUM_ACTIONS = 3
END_GAME, ROTATE,TURN, = list(range(NUM_ACTIONS))

class ActionLookup(Enum):
    END_GAME = "EndGame"
    ROTATE = "Rotate"

