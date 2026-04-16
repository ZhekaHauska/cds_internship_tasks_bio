import sys
import os
import numpy as np
from typing import Union, Any
from my_lib.base import BaseRunner
from my_lib.utils import read_config
from my_lib.metrics import AimLogger


class TestRunner(BaseRunner):
    def make_agent(self, agent_type, conf):
        if agent_type == 'random':
            from my_lib.agents import RandomAgent
            agent = RandomAgent(**conf)
        else:
            raise NotImplementedError
        return agent

    @staticmethod
    def make_environment(env_type, conf, setup):
        if env_type == 'gridworld':
            from my_lib.envs import GridWorldWrapper
            env = GridWorldWrapper(conf, setup)
        else:
            raise NotImplementedError
        return env

    def switch_strategy(self, strategy):
        if strategy == 'random':
            self.reward_free = True
        elif strategy == 'non-random':
            self.reward_free = False

    @property
    def state(self):
        env = self.environment.environment
        r, c = env.r, env.c
        return r * env.w + c

    @property
    def state_visited(self):
        env = self.environment.environment
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        values[r, c] = 1
        return values, 1


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config: dict[str, Union[Union[dict[str, Any], list[Any]], Any]] = dict()

    # main part
    config['run'] = read_config(config_path)

    env_conf_path = config['run'].pop('env_conf')
    config['env_type'] = env_conf_path.split('/')[-2]
    config['env'] = read_config(env_conf_path)

    agent_conf_path = config['run'].pop('agent_conf')
    config['agent_type'] = agent_conf_path.split('/')[-2]
    config['agent'] = read_config(agent_conf_path)

    metrics_conf = config['run'].pop('metrics_conf')
    if metrics_conf is not None:
        config['metrics'] = read_config(metrics_conf)

    if config['run']['seed'] is None:
        config['run']['seed'] = int.from_bytes(os.urandom(4), 'big')

    logger = config['run'].pop('logger')
    if logger is not None:
        if logger == 'aim':
            logger = AimLogger(config)
        else:
            raise NotImplementedError

    runner = TestRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/gridworld_random.yaml'
    main(os.environ.get('RUN_CONF', default_config))
