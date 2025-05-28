import random
import numpy as np
import torch
from config.base_config import BaseConfig

from builder import Builder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    baseconfig = BaseConfig()

    # set the global seed
    set_seed(baseconfig.config["yaml-config"]["env"]["seed"])

    # Builder(baseconfig).build().train(start_ep=0)
    Builder(baseconfig).build().train()

    # # test
    # deepscale = Builder(baseconfig).build()
    # deepscale.agent.load("DeepScale", "Autoscaling-v1", 80)
    # total_score, average_respones, total_costs = deepscale.eval()
    # print(total_score, average_respones, total_costs)

if __name__ == "__main__":
    main()
