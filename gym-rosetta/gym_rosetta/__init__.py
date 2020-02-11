from gym.envs.registration import register

register(
    id='protein-fold-v0',
    entry_point='gym_rosetta.envs:ProteinFoldEnv',
)