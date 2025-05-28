import argparse
import datetime

curr_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")


parser = argparse.ArgumentParser(description='Arguments for ppo_dws')
# args for device
parser.add_argument('--device', type=str, default="cpu", help='Processor type')
parser.add_argument('--jobs', type=str, default=8, help='Parrallel jobs')
parser.add_argument('--save_freq', type=int, default=2, help='save the model')
parser.add_argument('--log_path', type=str, default=f"./logs/test/")
parser.add_argument('--model_path', type=str, default=f"./logs/test/model/")

# args for Env
parser.add_argument('--seed', type=int, default=0, help='Seed of the environment')
parser.add_argument('--app_num', type=int, default=1, help='Number of workflows within a problem instance')
parser.add_argument('--app_size', type=str, default='A13', help='Workflow patterns within a problem instance, i.e., S, M, L, XL')
parser.add_argument('--env_name', type=str, default='Autoscaling-v1', help='Name of environment')
parser.add_argument('--vm_types', type=int, default=5, help='Number of VM types')
parser.add_argument('--workload_pattern', type=int, default=0, help='NASA or WIKI')
parser.add_argument('--budget', type=int, default=200, help='budget')
parser.add_argument('--scale', type=int, default=2, help='scale')
parser.add_argument("--time_slot", type=int, default=180, help="time slot of each decision")

# args for HGAT policy
parser.add_argument('--heads', type=int, default=1, help='heads of GAT')
parser.add_argument('--gat_hidden_layer', type=int, default=64, help='hidden layer of GAT')
parser.add_argument('--in_channels_pm', type=int, default=2, help='input size of PM layer')
parser.add_argument('--in_channels_vm', type=int, default=5, help='input size of VM layer')
parser.add_argument('--in_channels_con', type=int, default=11, help='input size of container layer')
parser.add_argument('--out_channels_con', type=int, default=8, help='output size of container layer')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden layer of MLP')
parser.add_argument('--output_dim', type=int, default=1, help='output size of container selection MLP')
parser.add_argument('--output_dim_scale', type=int, default=1, help='output size of container scaling MLP')
parser.add_argument('--policy_id', type=int, default=-1, help='')

# args for ES-RL
parser.add_argument('--sigma_init', type=float, default=0.05, help='Sigma init: noise standard deviation')
parser.add_argument('--sigma_decay', type=float, default=0.999, help='decay ratio of Sigma init')
parser.add_argument('--population_size', type=int, default=40, help='Population size') 
parser.add_argument('--generation_num', type=int, default=3000, help='Iteration times of GP')
parser.add_argument('--learning_rate', type=int, default=0.01, help='Iteration times of GP')
parser.add_argument('--optimizer', type=str, default="SGD", help='Adam or SGD')

# DQN
parser.add_argument('--max_ep', type=int, default=200, help='max number of episodes')
parser.add_argument('--state_dim', type=int, default=6, help='state dimensions')
parser.add_argument('--action_dim', type=int, default=33, help='action dimensions')
parser.add_argument('--net_width', type=int, default=64, help='width of MLP')
parser.add_argument('--dvc', type=str, default='mps', help='device')
parser.add_argument('--noise_net', type=bool, default=False, help='whether use noise net')
parser.add_argument('--Duel', type=bool, default=True, help='whether use duel net')
parser.add_argument('--Double', type=bool, default=True, help='whether use double net')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--buffer_size', type=int, default=100000, help='buffer size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--exp_noise', type=float, default=0.2, help='noise for exploration')
parser.add_argument('--holistc', type=bool, default=True, help='whether scale holistic')
parser.add_argument('--gamma', type=float, default=1)

configs = parser.parse_args() 
