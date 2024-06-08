from environment.loader import env_from_config

from utils import TensorboardCallback,write_json,create_policy_paths

from stable_baselines3 import SAC,PPO,DDPG
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import os

available_policy = {"SAC":SAC,"PPO":PPO,"DDPG":DDPG,"RecurrentPPO":RecurrentPPO}

from config.trainRL_config import RL_test

CONFIG = RL_test
LOG_DIR = "runs/RL"
SAVE_PATH = "RLmodel"
RENDER = False

SAVE_PATH,LOG_DIR = create_policy_paths(SAVE_PATH,LOG_DIR,CONFIG['algorithm']['policy'])

# set up logger
new_logger = configure(LOG_DIR, ["tensorboard"]) # ["stdout", "csv", "tensorboard"]

env = env_from_config(CONFIG,RENDER=RENDER)

try:

    Policy = available_policy[CONFIG["algorithm"]["policy"]]
    model = Policy('MlpPolicy', 
                env, 
                verbose=1, 
                seed=CONFIG['algorithm']['seed'], 
                # tensorboard_log=LOG_DIR,
                    device='cuda',
                **CONFIG['algorithm']['model_config'])

    model.set_logger(new_logger)

    write_json(CONFIG,SAVE_PATH)
    model.learn(total_timesteps=CONFIG['train_config']['total_timesteps'],
                callback=[ TensorboardCallback(1), CheckpointCallback(
                    save_freq=CONFIG['train_config']['total_timesteps'] // CONFIG['train_config']["num_checkpoints"],
                    save_path=SAVE_PATH,
                    name_prefix="model")], reset_num_timesteps=False)
    
except Exception as e:
    print(e)

finally:
    env.close()