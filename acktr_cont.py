import numpy as np
import tensorflow as tf
from baselines import logger
import baselines.common as common
from baselines.common import tf_util as U
from baselines.acktr import kfac
# from baselines.acktr.filters import ZFilter
from baselines.common.filters import ZFilter
import os
import random
import magroboenv.EnvUtils as Utils

from DataGen import Bfield

import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def pathlength(path):
    return path["reward"].shape[0]# Loss function that we'll differentiate to get the policy gradient

def incAngle(x):
    while(True):
        x += round(random.uniform(-180, 180), 2)
        if x > -180 and x < 180:
            break
        else:
            x = 0
    return x


def rollout(env, policy, max_pathlength, animate=False, obfilter=None):
    """
    Simulate the env and policy for max_pathlength steps
    """
    init_ob = env.reset()
    # prev_ob = np.float32(np.zeros(init_ob.shape))
    zero_ob = np.array([0.0 for _ in range(6)])
    init_ob = np.concatenate((zero_ob, init_ob[6:]), axis=0)
    print("Intial Observation: ", init_ob)

    if not obfilter == None:
        init_ob = obfilter(init_ob)
    terminated = False

    terminal_rewards = []

    obs = []
    acs = []
    ac_dists = []
    logps = []
    rewards = []
    last_ac = np.array([])
    for _ in range(max_pathlength):
        if animate:
            env.render()
        # state = np.concatenate([init_ob[:6], init_ob[6:]], -1)
        state = np.array(init_ob)
        obs.append(state)
        ac, ac_dist, logp = policy.act(state)
        # print(type(ac))
        assert(not np.array_equal(ac, last_ac))
        last_ac = ac
        acs.append(ac)
        ac_dists.append(ac_dist)
        logps.append(logp)
        scaled_ac = env.action_space.low + (ac + 1) * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
        ob, rew, done, info = env.step(scaled_ac)
        if obfilter: ob = obfilter(ob)
        rewards.append(rew)
        terminal_rewards.append(rew)

        # goal_config, last_goal_config, slave_config, last_slave_config = env.get_all_configs()
        info = info[0]
        force_terminate = info[1]
        assert(len(info)==4, "Info length not 4")
        goal_config = info[0]
        last_goal_config= info[1]
        slave_config = info[2]
        last_slave_config = info[3]
        for g in range(100):
            new_goal_config = goal_config 
            new_goal_config[3] = incAngle(new_goal_config[3])
            new_goal_config[4] = incAngle(new_goal_config[4])
            percent_error, new_rew = Utils.calculate_reward(new_goal_config, slave_config, last_slave_config)
            # new_ob = slave_config + new_goal_config
            new_ob = np.concatenate((zero_ob, new_goal_config), axis=0)
            if obfilter: new_ob = obfilter(new_ob)
            # new_state = np.concatenate([new_ob[:6], new_ob[6:]], -1)
            new_state = np.array(new_ob)
            
            obs.append(new_state)
            rewards.append(new_rew)
            acs.append(ac)
            ac_dists.append(ac_dist)
            logps.append(logp)

        if done:
            terminated = True
            break
        # elif force_terminate:
        #     # Bias the data
        #     percent_error, new_rew = Utils.calculate_reward(goal_config, goal_config, last_slave_config)
        #     new_ob = goal_config + goal_config
        #     if obfilter: new_ob = obfilter(new_ob)
        #     # new_state = np.concatenate([new_ob[:6], new_ob[6:]], -1)
        #     new_state = np.array(new_ob)

        #     ac = Bfield.getCurrents(goal_config[:3], goal_config[3:])

        #     scaled_ac = env.action_space.low + (ac + 1) * (env.action_space.high - env.action_space.low)

        #     scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
        #     ob, rew, done, info = env.step(scaled_ac)
            
        #     obs.append(new_state)
        #     rewards.append(new_rew)
        #     acs.append(ac)
        #     ac_dists.append(ac_dist)
        #     logps.append(logp)


        prev_ob = np.copy(ob)

    return {"observation" : np.array(obs), "terminated" : terminated,
            "reward" : np.array(rewards), "action" : np.array(acs),
            "action_dist": np.array(ac_dists), "logp" : np.array(logps)}, terminal_rewards

def learn(env, policy, vf, gamma, lam, timesteps_per_batch, num_timesteps,
    animate=False, callback=None, desired_kl=0.002, fname='./training.ckpt'):

    mean_logger = setup_logger("Mean Logger", "log/episode_mean.txt")

    # print("Filter shape:  ", env.observation_space.shape)
    space = (env.observation_space.shape[0]*2,)
    obfilter = ZFilter(space)

    max_pathlength = env.spec.timestep_limit
    stepsize = tf.Variable(initial_value=np.float32(np.array(0.03)), name='stepsize') #0.03
    inputs, loss, loss_sampled = policy.update_info
    optim = kfac.KfacOptimizer(learning_rate=stepsize, cold_lr=stepsize*(1-0.9), momentum=0.9, kfac_update=2,\
                                epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                                weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    pi_var_list = []
    for var in tf.trainable_variables():
        if "pi" in var.name:
            pi_var_list.append(var)

    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_var_list)
    do_update = U.function(inputs, update_op)
    U.initialize()

    #changes
    if fname != None and tf.train.checkpoint_exists(fname):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), fname)
        logger.log("Model loaded from file {}".format(fname))
        
    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for qr in [q_runner, vf.q_runner]:
        assert (qr != None, "QR is None")
        enqueue_threads.extend(qr.create_threads(tf.get_default_session(), coord=coord, start=True))

    i = 0
    timesteps_so_far = 0
    total_reward = float()
    while True:
        print ("Timestep Number: %d of %d" % (timesteps_so_far, num_timesteps))
        if timesteps_so_far > num_timesteps:
            break
        logger.log("********** Iteration %i ************"%i)

        #Save model every 100 iterations
        if fname != None and (i % 100 == 0):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.train.Saver()
            saver.save(tf.get_default_session(), fname)
            logger.log("Model saved to file {}".format(fname))
            env.seed()
            
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        terminal_rew = []
        while True:
            path, temp_rew = rollout(env, policy, max_pathlength, animate=(len(paths)==0 and (i % 10 == 0) and animate), obfilter=obfilter)
            paths.append(path)
            terminal_rew.append(np.array(temp_rew))
            n = pathlength(path)
            timesteps_this_batch += n
            if timesteps_this_batch > timesteps_per_batch:
                break
        timesteps_so_far += 1

        # Estimate advantage function
        vtargs = []
        advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            vtargs.append(return_t)
            vpred_t = vf.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = rew_t + gamma*vpred_t[1:] - vpred_t[:-1]
            adv_t = common.discount(delta_t, gamma * lam)
            advs.append(adv_t)
        # Update value function
        vf.fit(paths, vtargs)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        oldac_dist = np.concatenate([path["action_dist"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        # Policy update
        do_update(ob_no, action_na, standardized_adv_n)

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)
        # Adjust stepsize
        kl = policy.compute_kl(ob_no, oldac_dist)
        if kl > desired_kl * 2:
            logger.log("kl too high")
            tf.assign(stepsize, tf.maximum(min_stepsize, stepsize / 1.5)).eval()
        elif kl < desired_kl / 2:
            logger.log("kl too low")
            tf.assign(stepsize, tf.minimum(max_stepsize, stepsize * 1.5)).eval()
        else:
            logger.log("kl just right!")


        terminal_rew = np.array(terminal_rew)
        rew_mean = np.mean([path.sum() for path in terminal_rew])
        rew_sem = np.std([path.sum()/np.sqrt(len(terminal_rew)) for path in terminal_rew])
        len_mean = np.mean([path.shape[0] for path in terminal_rew])

        # rewList = []
        # for path in paths:
        #     trew = []
        #     rew_i = 0
        #     while True:
        #         trew.append(path["reward"][rew_i])
        #         rew_i += 11
        #         if rew_i > (len(path["reward"])-1):
        #             break
        #     rewList.append( np.array(trew) )
        # rewList = np.array(rewList)

        # rew_mean = np.mean([path.sum() for path in rewList])
        # rew_sem = np.std([path.sum()/np.sqrt(len(rewList)) for path in rewList])
        # len_mean = np.mean([path.shape[0] for path in rewList])

        # rew_mean = np.mean([path["reward"].sum() for path in paths])
        # rew_sem = np.std([path["reward"].sum()/np.sqrt(len(paths)) for path in paths])
        # len_mean = np.mean([pathlength(path) for path in paths])

        total_reward += rew_mean

        logger.record_tabular("EpRewMean", rew_mean)
        logger.record_tabular("EpRewSEM", rew_sem)
        logger.record_tabular("EpLenMean", len_mean)
        logger.record_tabular("TotalRewardMean", total_reward)
        logger.record_tabular("KL", kl)
        if callback:
            callback()
        logger.dump_tabular()

        mean_logger.info("Result for episode {}  of {}: Sum: {}, Average: {}, Length: {}".format(timesteps_so_far, num_timesteps, rew_mean, rew_sem, len_mean))

        i += 1

    if fname != None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), fname)
        logger.log("Model saved to file {}".format(fname))
        env.seed()
    coord.request_stop()
    coord.join(enqueue_threads)

