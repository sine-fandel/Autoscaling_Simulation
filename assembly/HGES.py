import multiprocessing as mp
import os
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pickle as pickle
import yaml
import sys

from assembly.base_assemble import BaseAssembleRL
from utils.running_mean_std import RunningMeanStd

from utils.policy_dict import agent_policy
import builder

from utils.torch_util import get_flatten_params, set_flatten_params

from copy import deepcopy


class HGES(BaseAssembleRL):

    def __init__(self, config, env, policy, optim):
        super(HGES, self).__init__()

        self.config = config

        self.env = env
        self.policy = policy
        self.optim = optim

        #  settings for running
        self.running_mstd = self.config.config['yaml-config']["optim"]['input_running_mean_std']
        if self.running_mstd:  # Init running mean and std
            self.ob_rms = RunningMeanStd(shape=self.env.observation_space.shape)
            self.ob_rms_mean = self.ob_rms.mean
            self.ob_rms_std = np.sqrt(self.ob_rms.var)
        else:
            self.ob_rms = None
            self.ob_rms_mean = None
            self.ob_rms_std = None

        self.generation_num = self.config.config['yaml-config']["optim"]['generation_num']
        self.processor_num = self.config.config['runtime-config']['processor_num']

        self.eval_ep_num = self.config.config['runtime-config']['eval_ep_num']

        # log settings
        self.log = self.config.config['runtime-config']['log']
        self.save_model_freq = self.config.config['runtime-config']['save_model_freq']
        self.save_mode_dir = None

    def train(self):
        if self.log:
            # Init log repository
            now = datetime.now()
            curr_time = now.strftime("%Y%m%d%H%M%S%f")
            dir_lst = []
            self.save_mode_dir = f"logs/{self.env.name}/{curr_time}"
            dir_lst.append(self.save_mode_dir)
            dir_lst.append(self.save_mode_dir + "/saved_models/")
            dir_lst.append(self.save_mode_dir + "/train_performance/")
            for _dir in dir_lst:
                os.makedirs(_dir)
            # shutil.copyfile(self.args.config, self.save_mode_dir + "/profile.yaml")
            # save the running YAML as profile.yaml in the log
            with open(self.save_mode_dir + "/profile.yaml", 'w') as file:
                yaml.dump(self.config.config['yaml-config'], file)
                file.close()

        # Start with a population init
        # init_path = "/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/logs/Autoscaling-v1/20241015120255376449/saved_models/ep_100.pt"
        # start_point = 100
        init_path = ""
        start_point = 0
        population = self.optim.init_population(self.policy, self.env, init_path)
        if self.config.config['yaml-config']['optim']['maximization']:
            best_reward_so_far = float("-inf")
        else:
            best_reward_so_far = float("inf")

        for g in range(start_point, self.generation_num):
            start_time = time.time()

            p = mp.Pool(self.processor_num)

            arguments = [(indi, self.env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                          self.processor_num, g, self.config, self.save_mode_dir) for indi in population]
            
            # start rollout works
            start_time_rollout = time.time()

            if self.processor_num > 1:
                results = p.map(worker_func, arguments)
            else:
                results = [worker_func(arg) for arg in arguments]

            p.close()

            # end rollout
            end_time_rollout = time.time() - start_time_rollout

            # start evol
            start_time_evol = time.time()
            results_df = pd.DataFrame(results).sort_values(by=['policy_id'])
            population, sigma_curr, best_reward_per_g = self.optim.next_population(self, results_df, g)
            end_time_evol = time.time() - start_time_evol

            end_time_generation = time.time() - start_time

            # update best reward so far
            if self.config.config['yaml-config']['optim']['maximization'] and (best_reward_per_g > best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            if (not self.config.config['yaml-config']['optim']['maximization']) and (
                    best_reward_per_g < best_reward_so_far):
                best_reward_so_far = best_reward_per_g

            # print runtime infor
            cur_log = f"\nepisode: {g}, best reward so far: {best_reward_so_far:.4f}, best reward of the current generation: {best_reward_per_g:.4f}, \
                    sigma: {sigma_curr:.3f}, time_generation: {end_time_generation:.2f}, rollout_time: {end_time_rollout:.2f}, \
                    rt: {results_df.iloc[0]['average_resptime']:.2f}, cost: {results_df.iloc[0]['VM_cost']:.2f}"
            print(
                cur_log, flush=True
            )

            # # update mean and std every generation
            # if self.running_mstd:
            #     hist_obs = []
            #     hist_obs.append(results_df['hist_obs'])
            #     # Update future ob_rms_mean  and  ob_rms_std
            #     self.ob_rms.update(hist_obs)
            #     self.ob_rms_mean = self.ob_rms.mean
            #     self.ob_rms_std = np.sqrt(self.ob_rms.var)

            if self.log:
                # if self.running_mstd:
                #     results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
                # return row of parent policy, i.e., policy_id = -1
                # results_df = results_df.loc[results_df['policy_id'] == -1]
                # with open(self.save_mode_dir + "/train_performance" + "/training_record.csv", "a") as f:
                #     results_df.to_csv(f, index=False, header=False)

                elite = self.optim.get_elite_model()
                if (g + 1) % self.save_model_freq == 0:
                    save_pth = self.save_mode_dir + "/saved_models" + f"/ep_{(g + 1)}.pt"
                    print(save_pth)
                    torch.save(elite.state_dict(), save_pth)
                    # if self.running_mstd:
                    #     save_pth = self.save_mode_dir + "/saved_models" + f"/ob_rms_{(g + 1)}.pickle"
                    #     f = open(save_pth, 'wb')
                    #     pickle.dump(np.concatenate((self.ob_rms_mean, self.ob_rms_std)), f,
                    #                 protocol=pickle.HIGHEST_PROTOCOL)
                    #     f.close()

    def eval(self):
        # load policy from log
        self.policy.load_state_dict(torch.load(self.config.config['runtime-config']['policy_path']))
        # create an individual wrapped with agent id
        indi = agent_policy(self.env.get_agent_ids(), self.policy)
        # load runtime mean and std
        if self.running_mstd:
            with open(self.config.config['runtime-config']['rms_path'], "rb") as f:
                ob_rms = pickle.load(f)
                self.ob_rms_mean = ob_rms[:int(0.5 * len(ob_rms))]
                self.ob_rms_std = ob_rms[int(0.5 * len(ob_rms)):]

        self.policy.eval()
        # use a random seed for simulator in testing setting
        g = np.random.randint(2 ** 31)

        arguments = [(indi, self.env, self.optim, self.eval_ep_num, self.ob_rms_mean, self.ob_rms_std,
                      self.processor_num, g, self.config)]

        results = [worker_func(arg) for arg in arguments]

        results_df = pd.DataFrame(results)

        if self.log:
            results_df = results_df.drop(['hist_obs'], axis=1)  # remove hist_obs from  log
            dir_test = os.path.dirname(self.config.config['runtime-config']['config']) + "/test_performance"
            if not os.path.exists(dir_test):
                os.makedirs(dir_test)
            results_df.to_csv(dir_test + "/testing_record.csv", index=False, header=False, mode='a')


def worker_func(arguments):
    indi, env, optim, eval_ep_num, ob_rms_mean, ob_rms_std, processor_num, g, config, save_mode_dir = arguments
    # print(indi, env, optim, eval_ep_num, ob_rms_mean, ob_rms_std, processor_num, g, config, save_mode_dir)
    # if processor_num > 1:
    #     env = builder.build_env(config.config)

    hist_rewards = {}  # rewards record all evals
    hist_obs = {}  # observation  record all evals
    hist_actions = {}
    obs = []
    total_reward = 0

    for ep_num in range(eval_ep_num):
        # makesure identical training instances for each ep_num over one generation
        # if ep_num == 0:
        #     states = env.reset(g)  # we also reset random.seed and np.random.seed in env.reset
        # else:
        #     seed = np.random.randint(2 ** 31)  # same random seed across indi
        #     states = env.reset(seed)
        states = env.reset(seed=g)

        rewards_per_eval = []  # for recording rewards for the current evaluation episode
        obs_per_eval = []      # for recording observations for the current evaluation episode
        actions_per_eval = []  # for recording actions for the current evaluation episode
        done = False
        while not done:
            actions = {}
            for agent_id, model in indi.items():
                s = states
                actions[agent_id] = model(s)
                states, r, done, _ = env.step(actions)
                rewards_per_eval.append(r)
                obs_per_eval.append(s)
                actions_per_eval.append(actions[agent_id])
                total_reward += r 

                # # trace observations
                # obs.append(states)

        hist_rewards[ep_num] = rewards_per_eval
        hist_obs[ep_num] = obs_per_eval
        hist_actions[ep_num] = actions_per_eval

    rewards_mean = total_reward / eval_ep_num

    if env.name in ["Autoscaling-v1"] and optim.name == "es_openai":
        if indi['0'].policy_id == -1:
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    "CON_execHour": env.episode_info["CON_execHour"],
                    "missDeadlineNum": env.episode_info["missDeadlineNum"],
                    "VM_cost": env.episode_info["VM_cost"],
                    "average_resptime": env.episode_info["average_resptime"],
                    "missDeadlineNum": env.episode_info["missDeadlineNum"]}
        else:  # we do not record detailed info for non-parent policy
            return {'policy_id': indi['0'].policy_id,
                    'rewards': rewards_mean,
                    "VM_execHour": np.nan,
                    "VM_totHour": np.nan,
                    "VM_cost": np.nan,
                    "SLA_penalty": np.nan,
                    "missDeadlineNum": np.nan}



    if ob_rms_mean is not None:
        return {'policy_id': indi['0'].policy_id, 'hist_obs': obs, 'rewards': rewards_mean}

    return {'policy_id': indi['0'].policy_id,
            'rewards': rewards_mean}

    # results_produce(env.name, optim.name)


def discount_rewards(rewards):
    gamma = 0.99  # gamma: discount factor in rl
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * gamma + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards
