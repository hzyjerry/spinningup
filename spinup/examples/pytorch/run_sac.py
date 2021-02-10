from copy import deepcopy
import itertools
import numpy as np
import torch
import gym
import upn.envs
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.algos.pytorch.sac.sac import *
from spinup.utils.logx import EpochLogger

model_path = ""

def env_fn():
    # return gym.make('ReacherIRL-v0')
    return gym.make('PointIRL-v0')

sac_upn(env_fn)