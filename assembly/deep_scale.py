"""
Tao's TSC 2023
DeepScale
"""


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

from assembly.base_assemble import BaseAssembleRL
from utils.running_mean_std import RunningMeanStd

from utils.policy_dict import agent_policy
from env.autoscaling_v1.simulator_as import ASEnv
from agent.DQN import DQN_agent

import copy



class DeepScale(BaseAssembleRL):

    def __init__(self, config, env: ASEnv, agent: DQN_agent):
        super(DeepScale, self).__init__()

        self.config = config

        self.env = env
        self.agent = agent

        
    def train(self, start_ep):
        current_ep = start_ep
        max_ep = 100
        if start_ep != 0:
            """
            start from checkpoint
            """
            # test_overall_return = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/test_returns.npy")[:start_ep]
            # test_costs = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/test_costs.npy")[:start_ep]
            # test_responses = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/test_responses.npy")[:start_ep]
            loss_stat = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/loss.npy")[:start_ep]
            overall_return = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/returns.npy")[:start_ep]
            responses = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/responses.npy")[:start_ep]
            costs = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/costs.npy")[:start_ep]
            total_steps = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/steps.npy")[0]
            self.agent.exp_noise = np.load("/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/exp_noise.npy")[0]
            self.agent.load("DeepScale", "Autoscaling-v1", start_ep)
        else:
            total_steps = 0
            overall_return = np.array([])
            aver_rewards = np.array([])
            responses = np.array([])
            costs = np.array([])
            test_overall_return = np.array([])
            test_aver_rewards = np.array([])
            test_responses = np.array([])
            test_costs = np.array([])
            loss_stat = np.array([])
        while current_ep < max_ep:
            s = self.env.reset(test=False)
            done = False
            current_ep += 1
            steps = 0
            returns = 0
            total_cost = 0
            total_responses = 0

            self.agent.exp_noise -= (0.2 - 0.01) / max_ep
            # print(self.agent.exp_noise)
            # print(total_steps)
            while not done:
                a = self.agent.select_action(state=s, deterministic=False)
                s_next, _, r, d, tr, c = self.env.step(steps, a)
                returns += np.power(0.99, steps) * r
                if d == True:
                    total_cost = c
                    average_responses = np.mean(tr)
    
                done = d

                # save experience in replay buffer
                self.agent.replay_buffer.add(s, a, r, s_next, d)
                s = s_next
                steps += 1
                total_steps += 1

                """
                training
                """
                if total_steps % 200 == 0:
                    update_tar = True
                else:
                    update_tar = False
                # if total_steps % 1000 == 0 and total_steps != 0: 
                #     if self.agent.exp_noise > 0.01:
                #         self.agent.exp_noise *= 0.99
                #     else:
                #         self.agent.exp_noise = 0.01
                # if total_steps % 50 == 0:
                    # for j in range(50):
                loss = self.agent.train(update_tar)
                loss_stat = np.append(loss_stat, loss)
                        
                if done == True:
                    overall_return = np.append(overall_return, returns)
                    responses = np.append(responses, average_responses)
                    costs = np.append(costs, total_cost)
                    # score, test_resp, test_co = self.eval()
                    # test_overall_return = np.append(test_overall_return, score)
                    # test_responses = np.append(test_responses, test_resp)
                    # test_costs = np.append(test_costs, test_co)
                    print(f"ep: {current_ep} return: {returns}; ART: {average_responses}; Cost: {total_cost} ")
                    np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/loss', loss_stat)
                    np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/returns', overall_return)
                    np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/responses', responses)
                    np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/costs', costs)
                    # np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/loss', loss_stat)
                    # np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/test_returns', test_overall_return)
                    # np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/test_responses', test_responses)
                    # np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/test_costs', test_costs)
                    # print(f"ep: {current_ep} reward: {total_reward}; ART: {average_responses / steps}; Cost: {total_cost / steps} ** ")
                    # print(f"ep: {current_ep} reward: {total_reward} ** ")

                    if current_ep % 10 == 0:
                        self.agent.save("DeepScale", self.env.name, current_ep)
                        np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/exp_noise', np.array([self.agent.exp_noise]))
                        np.save('/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/AutoScale/models/steps', np.array([total_steps]))

    def eval(self):
        s = self.env.reset(test=True)
        done = False
        total_score = 0
        total_respones = 0
        total_costs = 0
        steps = 0
        while not done:
            a = self.agent.select_action(s, deterministic=True)
            s_next, _, r, d, tr, c = self.env.step(steps, a)
            done = d
            total_score += np.power(0.99, steps) * r
            if d == True:
                total_costs = c
                average_respones = np.mean(tr)
            s = s_next

            steps += 1

        return total_score, average_respones, total_costs