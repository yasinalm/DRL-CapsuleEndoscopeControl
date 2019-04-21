import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
from time import sleep
import logging
from datetime import datetime
import sys

import magroboenv.myconfig as myconfig
import magroboenv.SimProbe as MProbe
import magroboenv.EnvUtils as Utils

def square(x):
    return x*x

def distance(nparray1, nparray2):
    sum = square(nparray1[0] - nparray2[0]) + square(nparray1[1] - nparray2[1]) + square(nparray1[2] - nparray2[2])
    return math.sqrt(sum)

class EnvSpec(object):
    def __init__(self, timestep_limit, id):
        self.timestep_limit = timestep_limit
        self.id = id
        
class MagRoboEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):

        date_str=datetime.now().strftime('%Y%m%d-%H%M%S')
        logfile=myconfig.Config.LOGFILE + date_str + ".log"

        logging.basicConfig(filename=logfile, level=logging.DEBUG)

        self.spec = EnvSpec(timestep_limit = myconfig.Config.TIMESTEP_LIMIT, id=1)

        # observation is the x, y, z coordinate & m. moments of the grid
        self.observation_space = spaces.Box(low=MProbe.MProbe.ob_low, high=MProbe.MProbe.ob_high)

	#Action Space => Current values
        if myconfig.Config.CURR_DEVIATE_ACTIVE == True:
            #self.action_space = spaces.Tuple(spaces.MultiDiscrete(MProbe.Current.deviate_action))
            self.action_space = spaces.Box(low=MProbe.Current.deviate_action_low, high=MProbe.Current.deviate_action_high, dtype=np.int8)
            # 0 -> no change; 1 -> +ve change; -1 -> -ve change
        else:
            self.action_space = spaces.Box(low=MProbe.Current.ac_low, high=MProbe.Current.ac_high)

        #initial condition
        self.state = None
        self.steps_beyond_done = None

        # simulation related variables
        self.seed()

        #        self.set_goal() inside reset()
        self.reset()
        #print(self.robot)

        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # print("ac={}".format(action))
        logging.debug("action={}".format(action))
        
        self._take_action(action)
        ob = self.state + MProbe.goal.read_sys_configuration()
        
        reward = self._get_reward()

        '''
        if reward == 4:
            done = True
        else:
            done = False'''
        done = False
        force_terminate = False
        if myconfig.Config.TRAINING_MODE == "COORD":
            if self.curr_dist >= 1.5*self.init_dist and self.count_ts >= myconfig.Config.RESET_STEP_COUNT:
                # print(" Reset Reward:{}, TS={}".format(reward, self.count_ts))
                done = False
            elif self.curr_dist < myconfig.Config.PROBE_DIM:
                # print(" Reset Goal Reward:{}, TS={}".format(reward, self.count_ts))
                done = True
            else:
                done = False
        elif myconfig.Config.TRAINING_MODE == "MOMENT":
            # if self.percentage_error > 120 and self.count_ts >= myconfig.Config.RESET_STEP_COUNT:
            if self.count_ts >= myconfig.Config.RESET_STEP_COUNT:
                # print(" Reset Reward:{}, TS={}".format(reward, self.count_ts))
                # reward *= (-10.0)
                done = True
                force_terminate = True
            elif self.percentage_error < 30:
                # print(" Reset Goal Reward:{}, TS={}".format(reward, self.count_ts))
                reward += (( 30.0 - self.percentage_error )/30.0)/2.0
                # done = True
            else:
                done = False
            # if self.curr_moment_dist >= 1.5*self.init_moment_dist and self.count_ts >= myconfig.Config.RESET_STEP_COUNT:
            #     print(" Reset Reward:{}, TS={}".format(reward, self.count_ts))
            #     done = True
            # elif self.curr_moment_dist < 0.01:
            #     print(" Reset Goal Reward:{}, TS={}".format(reward, self.count_ts))
            #     done = True
            # else:
            #     done = False

        
        info = self.get_all_configs()
        
        return ob, reward, done, {0:list(info), 1:force_terminate}

    def _take_action(self, action):

        for i in range(9):
            if math.isnan(action[i]):
                self.seed(0)
                return

        #change the current
        #print("Taking actions: ", str(action))
        if myconfig.Config.CURR_DEVIATE_ACTIVE == True:
            MProbe.desired_current.set_all_sys_curr_deviate(action)
        else:
            MProbe.desired_current.set_all_sys_current(action)

        #sleep for sometime before reading
        # sleep_time = 1.0 / myconfig.Config.RUN_TIMES_PER_SEC
        # sleep(sleep_time) #sleep in seconds
        
        # set action
        # MProbe.desired_current.set_all_sys_current(action)

        #read the changed orientation
        # self.state = MProbe.slave.read_sys_configuration()

        MProbe.slave.simulate_currents(MProbe.desired_current.read_sys_current())
        self.state = MProbe.slave.read_sys_configuration()

        self.count_ts += 1

    def reset(self):

        #read current orientation
        MProbe.slave.refresh()
        self.state = MProbe.slave.read_sys_configuration()

        #set goal
        self.set_goal()
        goal = MProbe.goal.read_sys_configuration()

        ob = self.state + goal 
        assert(len(ob) == 12, "goal length not 12")

        #generate random seed
        self.seed()

        self.count_ts = 0

        #Find Distance b/w start & goal
        self.init_dist = MProbe.slave.find_distance(MProbe.goal)
        self.init_moment_dist = MProbe.slave.find_moment_distance(MProbe.goal)
        self.reward_metrix = []
        nth_dist = self.init_dist / myconfig.Config.REWARD_GRADIENT
        nth_moment_dist = self.init_moment_dist / myconfig.Config.REWARD_GRADIENT

        if myconfig.Config.TRAINING_MODE == "COORD":
            for i in range(myconfig.Config.REWARD_GRADIENT):
                self.reward_metrix.append(i*nth_dist)
        elif myconfig.Config.TRAINING_MODE == "MOMENT":
            for i in range(myconfig.Config.REWARD_GRADIENT):
                self.reward_metrix.append(i*nth_moment_dist)
        else:
            sys.exit("Mode not found")
        
        self.curr_dist = self.init_dist
        self.curr_moment_dist = self.init_moment_dist
        
        return np.array(ob)

    def set_goal(self):
        MProbe.goal.refresh()

        # Setting output position to same as slave
        # Because of Bfield simulation setting
        slave_config = MProbe.slave.get_config()
        MProbe.goal.set_x(slave_config[0])
        MProbe.goal.set_y(slave_config[1])
        MProbe.goal.set_z(slave_config[2])

    def get_all_configs(self):
        goal_config = MProbe.goal.get_config()
        last_goal_config = MProbe.goal.get_last_config()
        slave_config = MProbe.slave.get_config()
        last_slave_config = MProbe.slave.get_last_config()

        return goal_config, last_goal_config, slave_config, last_slave_config

    def _get_reward(self):

        # self.last_dist = self.curr_dist
        # self.last_moment_dist = self.curr_moment_dist

        # self.curr_dist = MProbe.slave.find_distance(MProbe.goal)
        # self.curr_moment_dist = MProbe.slave.find_moment_distance(MProbe.goal)
        # print("eucledian distance: {} {}".format(self.curr_dist, self.curr_moment_dist))
        #print("goal: ({}, {}, {})".format(MProbe.goal.coordinate.x, MProbe.goal.coordinate.y, MProbe.goal.coordinate.z))
        # logging.debug("distance:{} {}".format(self.curr_dist, self.curr_moment_dist))

        goal_config, last_goal_config, slave_config, last_slave_config = self.get_all_configs()

        assert(not np.array_equal(goal_config, last_goal_config), "Unequal goal config")
        assert(not np.array_equal(slave_config, last_slave_config), "Unequal slave config")

        self.percentage_error, rew = Utils.calculate_reward(goal_config, slave_config, last_slave_config)
        return rew
        
        
        # """ Reward is given for XYZ. """
        # if myconfig.Config.TRAINING_MODE == "COORD":
        #     print("Eucledian Distance = {}, Error={}%".format(self.curr_dist, (1.0 - self.curr_dist)/100.0 ))
        #     if self.curr_dist == 0.0:
        #         return 1
        #     elif self.curr_dist < myconfig.Config.PROBE_DIM:
        #         return 1
        #     else:
        #         for i in range(myconfig.Config.REWARD_GRADIENT):
        #             if self.curr_dist < self.reward_metrix[i]:
        #                 return i*0.1
        # elif myconfig.Config.TRAINING_MODE == "MOMENT":
        #     # goal_dist = MProbe.goal.find_moment_distance_xyz(0.0,0.0,0.0)
        #     # slave_dist = MProbe.slave.find_moment_distance_xyz(0.0,0.0,0.0)
        #     # last_dist = MProbe.slave.find_last_moment_distance_xyz(0.0,0.0,0.0)
        #     # self.percentage_error = abs(abs(goal_dist-slave_dist)*100)/abs(goal_dist-last_dist)

        #     num = MProbe.goal.find_moment_distance(MProbe.slave)
        #     last_moment = MProbe.slave.get_last_moment()
        #     dnum = MProbe.goal.find_moment_distance_xyz(last_moment.mx, last_moment.my, last_moment.mz)

        #     self.percentage_error = 100.0*num/dnum
        #     print("Eucledian Distance = {}, Error={}%".format(self.curr_moment_dist, self.percentage_error ))

        #     if self.percentage_error > 100:
        #         return 0
        #     else:
        #         return (1.0 - self.percentage_error/100)

        #     # if self.curr_moment_dist < 1:
        #     #     print ("Current Moment Dist: {}".format(self.curr_moment_dist))
        #     #     return 1
        #     # else:
        #     #     for i in range(myconfig.Config.REWARD_GRADIENT): #=10
        #     #         sum_reward = 0
        #     #         if self.curr_moment_dist < self.reward_metrix[i]:
        #     #             sum_reward += i*10
        #     #     return sum_reward
        # else:
        #     #TODO
        #     pass
        
        #return np.random.normal(self.curr_dist, 0.5)*10
        # return -1

    def render(self, mode='human', close=False):
        pass
    
    def close(self):
        pass
    

    
