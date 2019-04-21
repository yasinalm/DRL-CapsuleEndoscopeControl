import tensorflow as tf
import numpy as np

from policies import GaussianMlpPolicy

from baselines.common import tf_util as U
from gym import error, spaces, utils
from math import *

def getAngle(x1,y1,x2,y2):
    s = x1*x2 + y1*y2
    v = x1*y2 - y1*x2

    l1 = sqrt(x1*x1 + y1*y1)
    l2 = sqrt(x2*x2 + y2*y2)
    pr = l1*l2

    if l1 < 10e-6 or l2 < 10e-6 or pr < 10e-5:
        return 0.0

    s /= pr
    alpha = acos(min(1.0, max(-1.0,s)))

    if v < 0.0:
        alpha *= -1.0
        
    return alpha 

from envs.gathering_env import *
from envs.pursuit_env import *

def make_env(env_name, visualization=False):
    if env_name == "pursuit":
        return PursuitGameEnv(visualization=visualization)
    if env_name == "gathering":
        return GatheringGameEnv(visualization=visualization)
    return None

    




