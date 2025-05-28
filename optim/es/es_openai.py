from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from optim.base_optim import BaseOptim
from utils.optimizers import Adam
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params

from collections import deque


class ESOpenAI(BaseOptim):
    def __init__(self, config):
        super(ESOpenAI, self).__init__()
        self.name = config["name"]
        self.sigma_init = config["sigma_init"]
        self.sigma_curr = self.sigma_init   # noise standard deviation
        self.sigma_decay = config["sigma_decay"]
        self.learning_rate = config["learning_rate"]
        self.reinforce_learning_rate = config["reinforce_learning_rate"]
        self.population_size = config["population_size"]
        self.reward_shaping = config['reward_shaping']
        self.reward_norm = config['reward_norm']

        self.epsilons = []  # save epsilons with respect to every model

        self.agent_ids = None
        self.mu_model = None        # policy network
        self.optimizer = None

    # Init policies of θ_t and (θ_t + σϵ_i)
    def init_population(self, policy: torch.nn.Module, env, init_path=""):
        # first, init θ_t
        self.agent_ids = env.get_agent_ids()
        if init_path == "":
            policy.norm_init()
        else:
            policy.restore_params(init_path)
        self.mu_model = policy
        self.optimizer = Adam(theta=get_flatten_params(self.mu_model)['params'], stepsize=self.learning_rate)
        self.re_optimizer = torch.optim.Adam(self.mu_model.parameters(), lr=self.reinforce_learning_rate)

        # second, init (θ_t + σϵ_i)
        perturbations = self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)
        return perturbations

    def init_perturbations(self, agent_ids: list, mu_model: torch.nn.Module, sigma, pop_size):
        """
        return a group of policy network with different parameters
        """
        perturbations = []  # policy F_i
        self.epsilons = []  # epsilons list

        # add mu model to perturbations for future evaluation
        perturbations.append(agent_policy(agent_ids, mu_model))

        # init eps as 0 (a trick for the implementation only)
        zero_eps = deepcopy(mu_model)
        # print("Parameters update2:")
        # for param in zero_eps.parameters():
        #     print(param.data)

        zero_eps.zero_init()
        zero_eps_param_lst = get_flatten_params(zero_eps)
        self.epsilons.append(zero_eps_param_lst['params'])

        # a loop of producing perturbed policy
        for _num in range(pop_size):
            perturbed_policy = deepcopy(mu_model)
            perturbed_policy.set_policy_id(_num)

            perturbed_policy_param_lst = get_flatten_params(perturbed_policy)  # θ_t
            epsilon = np.random.normal(size=perturbed_policy_param_lst['params'].shape)  # ϵ_i
            perturbed_policy_param_updated = perturbed_policy_param_lst['params'] + epsilon * sigma  # θ_t + σϵ_i  

            set_flatten_params(perturbed_policy_param_updated, perturbed_policy_param_lst['lengths'], perturbed_policy)

            perturbations.append(agent_policy(agent_ids, perturbed_policy))
            self.epsilons.append(epsilon)  # append epsilon for current generation

        return perturbations

    def next_population(self, assemble, results, g):
        rewards = results['rewards'].tolist()
        best_reward_sofar = max(rewards)
        rewards = np.array(rewards)

        # fitness shaping
        if self.reward_shaping:
            rewards = compute_centered_ranks(rewards)

        # normalization
        if self.reward_norm:
            r_std = rewards.std()
            rewards = (rewards - rewards.mean()) / r_std


        # - 1 / n * sigma
        update_factor = 1 / ((len(self.epsilons) - 1) * self.sigma_curr)  # epsilon -1 because parent policy is included
        update_factor *= -1.0  # adapt to minimization

        # sum of (- 1 / n * sigma) * (F_j * epsilon_j)
        grad_param_list = np.sum(np.array(self.epsilons) * rewards.reshape(rewards.shape[0], 1), axis=0)
        grad_param_list *= update_factor    # new parameters

        flatten_weights = self.optimizer.update(grad_param_list) 
        # set the updated parameters for the policy network
        set_flatten_params(flatten_weights, get_flatten_params(self.mu_model)['lengths'], self.mu_model)

        # Print parameters before the update
        # print("Parameters before update1:")
        # for param in self.mu_model.parameters():
        #     print(param.data)

        # ################################################################################# Newly added
        # if g >= 1:

        # # all_episode_rewards = deque(maxlen=10)  # To store recent rewards for baseline calculation
        #     all_episode_rewards = deque(maxlen=5)  # rewards over five episodes, with each reward representing the total for an episode
        #     all_actions = []  # all actions from 5 sampled episodes
        #     all_states = np.empty((0, 8))  # all states from 5 sampled episodes
        #     all_len_state = []  # all states length from 5 sampled episodes

        #     # Run 5 episodes to collect rewards and calculate policy losses
        #     for _ in range(5):
        #         seed = g
        #         episode_rewards, saved_log_probs, actions, states, len_state = run_episode_for_reinforce(self.mu_model, assemble.env, seed, assemble.ob_rms_mean, assemble.ob_rms_std)

        #         all_actions.append(actions)
        #         all_states = np.concatenate((all_states, states), axis = 0)
        #         all_episode_rewards.append(sum(episode_rewards))
        #         all_len_state. append(len_state)

        #     # Calculate Rbar using the mean of all_episode_rewards
        #     Rbar = np.mean(all_episode_rewards)


        #     log_prob_re = reinforce_alg(self.mu_model, all_states, all_actions, all_len_state)

        #     # calculate policy loss
        #     log_prob_re = [log_prob_re[i:i + len(all_actions[0])] for i in range(0, len(log_prob_re), len(all_actions[0]))]
        #     # print(log_prob_re)
        #     episode_policy_loss = self.calculate_policy_loss(all_episode_rewards, log_prob_re, Rbar)

        #     # Update the policy using the average of the collected episode policy losses
        #     self.update_policy(episode_policy_loss)

        #     # Print parameters after the update
        #     # print("Parameters after update:")
        #     # for param in self.mu_model.parameters():
        #     #     print(param.data)
        #     ################################################################################ finish the new added part


        # Continue with the generation of new perturbations
        perturbations = self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)


        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return perturbations, self.sigma_curr, best_reward_sofar

    ######################################################################### newly added function
    def calculate_policy_loss(self, all_episode_rewards, log_probability, Rbar):
        list_policy_loss = []
        std_dev = np.std(all_episode_rewards) + eps

        for log_p, reward in zip(log_probability, all_episode_rewards):
            # Ensure log_p is a tensor with requires_grad
            if not isinstance(log_p, torch.Tensor):
                log_p = torch.tensor(log_p, dtype=torch.float, requires_grad=True)
            else:
                log_p = log_p.float().requires_grad_()


            total = sum([-c * (reward - Rbar)/std_dev for c in log_p])

            list_policy_loss.append(total)

        return list_policy_loss

######################################################################### newly added function
    def update_policy(self, episode_policy_loss):
        # Convert all elements in episode_policy_loss to tensors if they are not already
        episode_policy_loss = [torch.tensor(loss) if not isinstance(loss, torch.Tensor) else loss for loss in
                               episode_policy_loss]

        # Stack and calculate the mean loss

        # Print parameters before the update
        # print("Parameters before update1:")
        # for param in self.mu_model.parameters():
        #     print(param.data)

        mean_loss = torch.mean(torch.stack(episode_policy_loss))
        self.mu_model.zero_grad()
        mean_loss.backward()
        self.re_optimizer.step()


        # Check that gradients are computed
        # print("Gradients:")
        # for param in self.mu_model.parameters():
        #     if param.grad is not None:
        #         print(param.grad)
        #     else:
        #         print("No gradient for this parameter")


    def get_elite_model(self):
        return self.mu_model


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y



######################################################################### newly added function
def run_episode_for_reinforce(policy, env, seed, ob_rms_mean, ob_rms_std):
    state = env.reset(seed)
    saved_log_probs = []  # log_probs for each step of an episode
    rewards = []  # rewards for each step of an episode
    actions = []  # action for taken at each step of an episode
    states = np.empty((0, 8))  # to save states
    len_state = []  # length of state for each step of an episode
    done = False


    while not done:

        if ob_rms_mean is not None and ob_rms_std is not None:
            # Clip the values of ob_rms_std
            ob_rms_std = np.clip(ob_rms_std, 1, 1000)

            # print("ob_rms_mean:", ob_rms_mean)
            # print("ob_rms_std:", ob_rms_std)
            normalized_state = (state['0']['state'] - ob_rms_mean) / ob_rms_std   # normalizing states
        else:
            normalized_state = state['0']['state']

        # Ensure state dimensionality
        if normalized_state.ndim < 2:
            normalized_state = normalized_state[np.newaxis, :]

        if "removeVM" in state: 
            action, log_prob = policy.forward_reinforce(normalized_state, state["removeVM"])
        else:
            action, log_prob = policy.forward_reinforce(normalized_state)


        actions.append(action)

        states = np.concatenate((states, normalized_state), axis=0)
        len_state.append(len(state['0']['state']))

        action = {'0': action}
        state, reward, done, _ = env.step(action)

        saved_log_probs.append(log_prob)
        rewards.append(reward)

    return rewards, saved_log_probs, actions, states, len_state

eps = np.finfo(np.float32).eps.item()

######################################################################### newly added function
def reinforce_alg(policy, states, actions, length):  # all selected actions and its log_probs
    log_probss = policy.forward_reinforce2(states, length, actions)
    return log_probss
