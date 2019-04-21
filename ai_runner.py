import os
import argparse
import sys
sys.path.insert(0, "C:/Users/TARIQUE/Desktop/openai/baselines")

#from game_utils import *
import tensorflow as tf
from baselines.common import tf_util as U
from baselines.common import set_global_seeds
from baselines.acktr.filters import ZFilter
from policies import GaussianMlpPolicy

from magroboenv.magroboenv import *
visualization = False

def make_env(env_name):
    if env_name == "MagRoboEnv-v0":
        return MagRoboEnv()
    return None

def run_game(fname, seed, envname):
    #env = make_env(envname, visualization=visualization)
    env = make_env(envname)
    obfilter = ZFilter(env.observation_space.shape)

    set_global_seeds(seed)
    env.seed(seed)

    with tf.device('/cpu:0'):
        with tf.Session(config=tf.ConfigProto()) as sess:
            ob_dim = env.observation_space.shape[0]
            ac_dim = env.action_space.shape[0]

            with tf.variable_scope("pi"):
                policy = GaussianMlpPolicy(ob_dim, ac_dim)

            saver = tf.train.Saver()
            saver.restore(sess, fname)
        
            ob = env.reset()
            ob = obfilter(ob)

            prev_ob = np.float32(np.zeros(ob.shape))
 
            try:
                while True:
                    state = np.concatenate([ob, prev_ob], -1)
                    ac, ac_dist, logp = policy.act(state)
                    prev_ob = np.copy(ob)
            
                    scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
                    scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
                    ob, rew, done, _ = env.step(scaled_ac)
                    ob = obfilter(ob)

                    if done == True:
                        break
    
                    if visualization == True:
                        env.render()
            except KeyboardInterrupt:
                env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run gathering game with AI')
    parser.add_argument('--fname', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='MagRoboEnv-v0', help="Name of the environment")
    args = parser.parse_args()

    run_game(fname=args.fname, seed=args.seed, envname=args.env)
