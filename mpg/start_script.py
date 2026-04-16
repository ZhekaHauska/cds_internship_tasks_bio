import wandb
import yaml
import os

from mylib.hmm_runner import HMMRunner

# specify you login in wandb, you should also authorize with token first, see wandb quick start documentation
os.environ['WANDB_ENTITY'] = 'your_login'

config_path = 'configs/hmm_runner.yaml'

if __name__ == '__main__':
    # load configs

    config = dict()

    with open(config_path, 'r') as file:
        config['run'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['hmm_conf'], 'r') as file:
        config['hmm'] = yaml.load(file, Loader=yaml.Loader)

    with open(config['run']['mpg_conf'], 'r') as file:
        config['mpg'] = yaml.load(file, Loader=yaml.Loader)

    if config['run']['log']:
        # start wandb logger
        logger = wandb.init(
            project=config['run']['project_name'], entity=os.environ['WANDB_ENTITY'],
            config=config
        )
    else:
        logger = None

    runner = HMMRunner(logger, config)
    runner.run()
