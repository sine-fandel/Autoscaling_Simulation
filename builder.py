from utils.utils import get_state_num, get_action_num, is_discrete_action, get_nn_output_num
from assembly.deep_scale import DeepScale
from assembly.HGES import HGES

class Builder:
    def __init__(self, baseconfig):
        """
        Config, environment, agent and optimization
        """
        self.config = baseconfig
        self.env = None
        self.agent = None
        self.optim = None

    def build(self):
        self.env = build_env(self.config.config)
        self.config.config['yaml-config']["agent"]["discrete_action"] = is_discrete_action(self.env)  # based on the environment, decide if the action space is discrete
        self.config.config['yaml-config']["agent"]["state_num"] = get_state_num(self.env)  # based on the environment, generate the state num to build agent
        self.config.config['yaml-config']["agent"]["action_num"] = get_nn_output_num(self.env)  # based on the environment, generate the action num to build agent
        self.agent = build_agent(self.config.config['yaml-config']["agent"])

        if self.config.config['yaml-config']["agent"]["name"] == "HGAT":
            optim = build_optim(self.config.config['yaml-config']["optim"])

            return HGES(self.config, self.env, self.agent, optim)
        else:
            return DeepScale(self.config, self.env, self.agent)


def build_env(config):
    env_name = config["yaml-config"]["env"]["name"]
    config["yaml-config"]["env"]["evalNum"] = config['runtime-config']['eval_ep_num']
    if env_name == "Autoscaling-v1":
        from env.autoscaling_v1.simulator_as import ASEnv
        return ASEnv(env_name, config["yaml-config"]["env"])
    else:
        raise AssertionError(f"{env_name} doesn't support in this simulation yet.")

def build_agent(config):
    agent_name = config["name"]
    """Build agent based on the config"""


def build_optim(config):
    optim_name = config["name"]
    if optim_name == "es_openai":
        from optim.es.es_openai import ESOpenAI
        return ESOpenAI(config)
    else:
        raise AssertionError(f"{optim_name} doesn't support, please specify supported a optim in yaml.")

