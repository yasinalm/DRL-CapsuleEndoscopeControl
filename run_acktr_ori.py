import argparse
import logging
import os
import tensorflow as tf
import gym
import sys
sys.path.insert(0, "C:/Users/TARIQUE/Desktop/openai/baselines")

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common import tf_util as U
from baselines import bench

from acktr_cont import learn

from policies import GaussianMlpPolicy

from baselines.acktr.value_functions import NeuralNetValueFunction

#from game_utils import *
from magroboenv.magroboenv import *

visualization = False

def make_env(env_name):
    if env_name == "MagRoboEnv-v0":
        return MagRoboEnv()
    return None

def train(num_timesteps, seed, env_name, fname):

    env = make_env(env_name)
    if env == None:
        logger.log("Empty environment")
        return
        
    env = bench.Monitor(env, logger.get_dir(),  allow_early_resets=True)
    set_global_seeds(seed)
    env.seed(seed)

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        print("Observation dim: ", ob_dim)

        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)
            # policy = build_policy(env, "mlp")

        try:
            learn(env, policy=policy, vf=vf,
                gamma=0.99, lam=0.97, timesteps_per_batch=100, #4500
                desired_kl=0.002,
                num_timesteps=num_timesteps, animate=visualization, fname=fname)
                
        except KeyboardInterrupt:
            if fname != None:
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                saver = tf.train.Saver()
                saver.save(tf.get_default_session(), fname)
                logger.log("Model saved to file {}".format(fname))
                pass

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--fname', type=str, default='./ori_train/train_ori.ckpt')
    parser.add_argument('--env', type=str, default='MagRoboEnv-v0')
    parser.add_argument('--num-timesteps', type=int, default=int(4e4)) #1e3
    args = parser.parse_args()
    log_dir="./log"
    logger.configure(dir=log_dir)
    
    train(num_timesteps=args.num_timesteps, seed=args.seed, env_name=args.env, fname=args.fname)
