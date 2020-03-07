from pdbtools.clean import pdbClean
from pdbtools.helper import cmdline
import os

# for file in os.listdir('protein_data/shortest'):
#     try:
#         f = open(os.path.join('protein_data/shortest',file), 'r')
#         print(file)
#         pdb = f.readlines()
#         f.close()
#
#         # print("Loading %s" % pdb_file)
#         pdb_id = 'file'
#         pdb = pdbClean(pdb, renumber_residues=True)
#
#         g = open(os.path.join('protein_data/shortest', file), "w")
#         g.writelines(pdb)
#         g.close()
#     except:
#         pass


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
from pyrosetta import PyMOLMover
import numpy as np
init()
#
pdb = pose_from_pdb(os.path.join('protein_data/short', '1y32.pdb'))
pdb_2 = pose_from_pdb(os.path.join('protein_data/short', '1y32.pdb'))
pdb_seq = pose_from_sequence(pdb_2.sequence())
pymover = PyMOLMover()
pymover.keep_history(True)
pymover.apply(pdb)
pymover_2 = PyMOLMover()

for i in range(pdb.total_residue()):
    pdb_seq.set_psi(i + 1, pdb.psi(i + 1))
    pdb_seq.set_phi(i + 1, pdb.phi(i + 1))
    pdb_seq.set_omega(i + 1, pdb.omega(i + 1))

# pdb_seq = pose_from_sequence(pdb.sequence())
R5N = AtomID(1, 6)
R5CA = AtomID(2, 6)
R5C = AtomID(3, 6)
pdb_2.conformation().set_bond_angle(R5N, R5CA, R5C,1.90)
print(pdb_seq.conformation().bond_angle(R5N, R5CA, R5C))
print(pdb.conformation().bond_angle(R5N, R5CA, R5C))
pymover.apply(pdb_2)

print(CA_rmsd(pdb_seq, pdb))
