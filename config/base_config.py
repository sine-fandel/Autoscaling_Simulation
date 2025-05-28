from abc import *
import argparse
import yaml
from distutils.util import strtobool


class BaseConfig(metaclass=ABCMeta):

    def __init__(self, *args):
        parser = argparse.ArgumentParser(description='Arguments')
        parser.add_argument('--config', type=str,
                            default='config/autoscaling_es_openai.yaml',
                            help='A config path for env, policy, and optim')
        parser.add_argument('--processor-num', type=int, default=5, help='Specify processor number for multiprocessing')
        parser.add_argument('--eval-ep-num', type=int, default=1, help='Set evaluation number per iteration')
        # Settings related to logsa
        parser.add_argument("--log", default=True, action="store_true", help="Use log")
        parser.add_argument('--save-model-freq', type=int, default=20, help='Save model every a few iterations')

        # Overwrite some common values in YAML with command-line options, if needed.
        parser.add_argument('--seed', type=int, default=None, help='Replace seed value in  YAML')
        parser.add_argument('--reward', type=int, default=None, help='Select reward option')
        parser.add_argument('--sigma-init', type=float, default=None, help='Sigma init: noise standard deviation')
        parser.add_argument('--learning-rate', type=float, default=None, help='Replace learning rate in YAML')
        parser.add_argument('--reinforce-learning-rate', type=float, default=None, help='Replace reinforce learning rate in YAML')

        args = parser.parse_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

            # Replace seed value if command-line options on seed is not None
            if args.seed is not None:
                config['env']['seed'] = args.seed

            if args.reward is not None:
                config['env']['reward'] = args.reward

            if args.sigma_init is not None:
                config['optim']['sigma_init'] = args.sigma_init

            if args.learning_rate is not None:
                config['optim']['learning_rate'] = args.learning_rate

            if args.reinforce_learning_rate is not None:
                config['optim']['reinforce_learning_rate'] = args.reinforce_learning_rate

        self.config = {}
        self.config["runtime-config"] = vars(args)
        self.config["yaml-config"] = config

       
