
import retro
import numpy as np


def build_env():
    """
    Build a new fresh environement for the atari space invaders game

    returns:
        The environment
        A one hot encoding of the possible actions
    """

    retro_env = retro.make(game='SpaceInvaders-Atari2600')

    # Build an one hot encoding of the actions
    actions = np.array(np.identity(
        retro_env.action_space.n, dtype=int).tolist())

    return retro_env, actions
