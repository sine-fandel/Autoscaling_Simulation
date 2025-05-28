from gym.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary
from env.gym_openAI.simulator_gym import GymEnv

import torch.nn.functional as F
from torch import nn
import torch
import math

from torch_geometric.data import Data
from itertools import combinations
import networkx as nx
import numpy as np
import copy

"""
Current supported space class:
Box: gym.spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)  -> Box(3,4)
     gym.spaces.Box(low=np.array([-1.0, -2.0]).astype(np.float32), high=np.array([2.0, 4.0]).astype(np.float32), dtype=np.float32)  -> Box(2,)
Dict: self.observation_space = gym.spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})
Discrete: a = gym.spaces.Discrete(n=3, start=-1), a.sample() -> one output
MultiBinary: An n-shape binary space. 
             self.observation_space = spaces.MultiBinary(n=5)
             self.observation_space.sample() -> array([0, 1, 0, 1, 0], dtype=int8)
MultiDiscrete: The multi-discrete action space consists of a series of discrete action spaces with different number of actions in each
             gym.spaces.MultiDiscrete(nvec=[ 5, 2, 2 ], dtype=np.int64)
"""

# todo: consider 2d observation input, isDiscreteAction: continous +discrete action

def get_space_shape(space):
    """
    Get the size of a given space.

    :param space: a class instance from gym.spaces
    """
    assert(space)
    if isinstance(space, Box):
        shape = space.shape
        if len(shape) == 0:
            return None
        elif len(shape) == 1:
            return shape[0]
        else:
            return shape
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, MultiDiscrete):
        if len(space.nvec) == 0:
            return None
        else:
            return len(space.nvec)
    elif isinstance(space, MultiBinary):
        return space.n
    elif isinstance(space, Dict):
        temp = None
        for i in space.keys():
            item = space[i]
            if temp is None:
                temp = get_space_shape(item)
            elif (isinstance(get_space_shape(item), int)):
                temp += get_space_shape(item)
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError


def get_state_num(env) -> int:
    """
    Get the number of state inputs for the policy.
    Used by 'builder.py' to pass the number of input nodes to policy initialization

    :param env: Environment to get the size of the observation space
    """
    if isinstance(env, GymEnv):
        observation_space = env.env.observation_space
    else:
        observation_space = env.observation_space
    return get_space_shape(observation_space)


def get_action_num(env) -> int:
    """
    Get the number of action inputs for the policy.
    Used by 'builder.py' to pass the number of output nodes to policy initialization

    :param env: Environment to get the size of the observation space
    """
    if isinstance(env, GymEnv):
        action_space = env.env.action_space
    else:
        action_space = env.action_space
    return get_space_shape(action_space)


def is_discrete_action(env) -> bool:
    """
    Check if the action is discrete
    Used by 'builder.py' for policy initialization
    Box: np.float32

    :param env: Environment to get the size of the observation space
    """
    if isinstance(env, GymEnv):
        space = env.env.action_space
    else:
        space = env.action_space

    if isinstance(space, Box):
        return False
    elif isinstance(space, Discrete) or isinstance(space, MultiBinary) or isinstance(space, MultiDiscrete):
        return True
    else:
        raise NotImplementedError


def get_nn_output_num(env) -> int:

    if env.name == "Autoscaling-v1":
        from env.autoscaling_v1.simulator_as import ASEnv

    if isinstance(env, GymEnv):
        space = env.env.action_space
    else:
        space = env.action_space
    if isinstance(env, ASEnv):
        return 1
    elif not is_discrete_action(env):
        return get_space_shape(space)
    else:
        assert (space)
        if isinstance(space, Discrete):
            return space.n
        else:
            raise NotImplementedError

class NoisyLinear(nn.Module):
    '''From https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/3.Rainbow_DQN/network.py'''
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters() # for mu and sigma
        self.reset_noise() # for epsilon

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)  # mul是对应元素相乘
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    

def graph_construct(pm_queues, vm_queues, con_queues: dict, G):
    """
    construct the graph based on current cloud environment
    return: raw features and edge of container, VM and PM layers
    """
    # print(con_queues)
    
    is_active = np.vectorize(lambda obj: obj.active)
    pm_active = pm_queues[is_active(pm_queues)]
    vm_active = vm_queues[is_active(vm_queues)]
    num_pm = len(pm_active)
    num_vm = len(vm_active)
    con_G = copy.deepcopy(G)

    pm_edges = list(combinations(range(num_pm), 2))
    vm_edges = list(combinations(range(num_vm), 2))

    """
    pm graph
    the order of raw features follows the order of pm_active
    """
    pm_edge_index = torch.tensor(pm_edges, dtype=torch.long).t().contiguous()
    pm_edge_index = torch.cat([pm_edge_index, pm_edge_index.flip(0)], dim=1)

    # get PM features
    batch_norm_pm = nn.BatchNorm1d(num_features=2)
    pm_raw_features = torch.empty((0, 2))
    pm_id = 0
    for pm in pm_active:
        pm_edge_index = torch.cat([pm_edge_index, torch.tensor([[pm_id], [pm_id]])], dim=1)
        pm_id += 1
        pm_vcpu = pm.vcpu
        pm_max_vcpu = pm.max_vcpu
        aver_resptime = pm.get_aver_resptime()
        raw_feature = torch.tensor([[1 - pm_vcpu / pm_max_vcpu, pm_max_vcpu]], dtype=torch.float32)
        # raw_feature = torch.tensor([[1 - pm_vcpu / pm_max_vcpu, aver_resptime]], dtype=torch.float32)
        pm_raw_features = torch.cat([pm_raw_features, raw_feature], dim=0)
    
    # pm_max = pm_raw_features.max(dim=0).values
    # pm_raw_features = (pm_raw_features) / pm_max

    """
    VM graph
    the order of raw features follows the order of vm_active
    """
    vm_edge_index = torch.tensor(vm_edges, dtype=torch.long).t().contiguous()
    vm_edge_index = torch.cat([vm_edge_index, vm_edge_index.flip(0)], dim=1)
    # add pm to vm arc
    for pm_id in range(num_pm):
        pm = pm_active[pm_id]
        vm_list = pm.vm_list
        for vm in vm_list:
            temp_vm_id = np.where(vm_active == vm)[0][0]
            pm_arc = torch.tensor([[num_vm + pm_id],
                                    [temp_vm_id]])
            vm_edge_index = torch.cat([vm_edge_index, pm_arc], dim=1)
    
    # get VM features
    vm_features_num = 5
    vm_raw_features = torch.empty((0, vm_features_num))
    batch_norm_vm = nn.BatchNorm1d(num_features=vm_features_num)
    vm_id = 0
    for vm in vm_active:
        vm_edge_index = torch.cat([vm_edge_index, torch.tensor([[vm_id], [vm_id]])], dim=1)
        vm_id += 1
        vm_vcpu = vm.vcpu
        vm_max_vcpu = vm.max_vcpu
        vm_price = vm.price
        total_resptime = vm.get_total_resptime()[0]
        aver_resptime = vm.get_aver_resptime()
        vm_rental = vm.get_rental()
        # raw_feature = torch.tensor([[1- vm_vcpu / vm_max_vcpu, vm_max_vcpu, vm_price, vm_rental]], dtype=torch.float32)
        raw_feature = torch.tensor([[1- vm_vcpu / vm_max_vcpu, vm_max_vcpu, vm_price, vm_rental, total_resptime]], dtype=torch.float32)
        # raw_feature = torch.tensor([[1- vm_vcpu / vm_max_vcpu, vm_max_vcpu, vm_price, vm_rental]], dtype=torch.float32)
        # raw_feature = torch.tensor([[1- vm_vcpu / vm_max_vcpu, vm_max_vcpu, vm_price, vm_rental, aver_resptime, total_resptime]], dtype=torch.float32)
        # raw_feature = torch.tensor([[1- vm_vcpu / vm_max_vcpu, vm_max_vcpu / 48, vm_price, vm_rental / 300, aver_resptime / 700]], dtype=torch.float32)
        vm_raw_features = torch.cat([vm_raw_features, raw_feature], dim=0)

    # vm_max = vm_raw_features.max(dim=0).values
    # vm_raw_features = (vm_raw_features) / vm_max

    """
    container graph
    the order of raw featurs follows the new id of containers from 0 to x, 
    and get from inverse_new_id_map
    """
    bfs_order = list(nx.bfs_tree(con_G, source="start"))
    new_id_map = {old_id: new_id - 1 for new_id, old_id in enumerate(bfs_order) if old_id != "start"}
    inverse_new_id_map = dict(zip(new_id_map.values(), new_id_map.keys()))  # use this dict to get raw feature of the new graph

    G_new = nx.DiGraph()

    for u, v in con_G.edges():
        if u != "start":
            G_new.add_edge(new_id_map[u], new_id_map[v])
            G_new.add_edge(new_id_map[v], new_id_map[u])
    
    con_new_id = G_new.nodes()
    con_edge_index = torch.tensor(list(G_new.edges), dtype=torch.long).t().contiguous()
    # # add vm to con arc
    # for vm_id in range(num_vm):
    #     vm = vm_active[vm_id]
    #     container_list = vm.container_list
    #     for con in container_list:
    #         vm_arc = torch.tensor([[len(con_new_id) + vm_id],
    #                                 [new_id_map[con.conid]]])
    #         con_edge_index = torch.cat([con_edge_index, vm_arc], dim=1)

    # get container features
    con_features_num = 11
    con_raw_features = torch.empty((0, con_features_num))
    batch_norm_con = nn.BatchNorm1d(num_features=con_features_num)
    for new_id in con_new_id:
        orig_id = inverse_new_id_map[new_id]
        # VM features
        host_vm = con_queues[orig_id].vm
        host_pm = con_queues[orig_id].pm
        host_pm_vcpu = host_pm.vcpu
        host_pm_max_vcpu = host_pm.max_vcpu
        host_vm_vcpu = host_vm.vcpu
        host_vm_max_vcpu = host_vm.max_vcpu
        host_vm_price = host_vm.price
        total_resptime = host_vm.get_total_resptime()[0]
        aver_resptime = host_vm.get_aver_resptime()
        host_vm_rental = host_vm.get_rental()

        con_vcpu = con_queues[orig_id].vcpu
        max_scal_vcpu = con_queues[orig_id].max_scal_vcpu
        # process_time = con_queues[orig_id].totalProcessTime
        degree = G_new.in_degree(new_id) + G_new.out_degree(new_id)
        pending_num = con_queues[orig_id].conQueue.qlen()
        # pending_task_time = con_queues[orig_id].pendingTaskTime
        aver_resptime = con_queues[orig_id].aver_resptime
        total_resptime = con_queues[orig_id].total_resptime
        workload_his = con_queues[orig_id].workload_his
        if len(workload_his) > 2:
            predicted_workload = (workload_his[-1] + workload_his[-2]) / 2
        else:
            predicted_workload = 0
        # violate_num = con_queues[orig_id].violate_num
        # raw_feature = torch.tensor([[con_vcpu, max_scal_vcpu, degree, pending_num]], dtype=torch.float32)
        # raw_feature = torch.tensor([[con_vcpu, max_scal_vcpu, degree, pending_num, aver_resptime]], dtype=torch.float32)
        # raw_feature = torch.tensor([[con_vcpu, max_scal_vcpu, degree, pending_num, aver_resptime, predicted_workload]], dtype=torch.float32)
        raw_feature = torch.tensor([[con_vcpu, max_scal_vcpu, degree, pending_num, aver_resptime, predicted_workload, \
                                    1 - host_vm_vcpu / host_vm_max_vcpu, host_vm_max_vcpu, host_vm_price, host_vm_rental, \
                                    1 - host_pm_vcpu / host_pm_max_vcpu]], dtype=torch.float32)

        # raw_feature = torch.tensor([[con_vcpu, max_scal_vcpu, degree, pending_num, predicted_workload]], dtype=torch.float32)
        # raw_feature = torch.tensor([[con_vcpu, max_scal_vcpu, degree, pending_num, total_resptime, aver_resptime]], dtype=torch.float32)
        # raw_feature = torch.tensor([[con_vcpu / 48, max_scal_vcpu / 48, degree / 10, pending_num / 150, aver_resptime / 700]], dtype=torch.float32)
        con_raw_features = torch.cat([con_raw_features, raw_feature], dim=0)

    # con_max = con_raw_features.max(dim=0).values
    # # con_std = con_raw_features.std(dim=0)
    # con_raw_features = (con_raw_features) / con_max

    return pm_raw_features, pm_edge_index, vm_raw_features, vm_edge_index, con_raw_features, con_edge_index, inverse_new_id_map

