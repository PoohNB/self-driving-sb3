from environment.loader import env_from_config

from utils import TensorboardCallback,write_json,write_pickle,create_policy_paths

from stable_baselines3 import SAC,PPO,DDPG
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import os

available_AlgorithmRL = {"SAC":SAC,"PPO":PPO,"DDPG":DDPG,"RecurrentPPO":RecurrentPPO}

from config.trainRL_config import RL_test,RL_test3,RL_rnnppo_1,RL_SAC_v2,RL_SAC_v2_con

reload_model = "RLmodel/SAC_5/model_100000_steps.zip"
CONFIG = RL_SAC_v2_con
LOG_DIR = "runs/RL"
SAVE_PATH = "RLmodel"
RENDER = True

algo_config = CONFIG['train']["algorithm"]
train_config = CONFIG['train']['train_config']

if algo_config['method'] not in available_AlgorithmRL:
    raise ValueError("Invalid algorithm name")

SAVE_PATH,LOG_DIR = create_policy_paths(SAVE_PATH,LOG_DIR,algo_config['method'])

env = env_from_config(CONFIG['env'],RENDER=RENDER)

try:
    new_logger = configure(LOG_DIR, ["tensorboard"])  # ["stdout", "csv", "tensorboard"]
    write_json(CONFIG,os.path.join(SAVE_PATH,"config.json"))
    write_pickle(CONFIG['env'],os.path.join(SAVE_PATH,"env_config.pkl"))

    AlgorithmRL = available_AlgorithmRL[algo_config["method"]]
    if reload_model == "":
        model = AlgorithmRL(env=env, 
                    verbose=1, 
                    seed=algo_config['seed'], 
                    # tensorboard_log=LOG_DIR,
                        device='cuda',
                    **algo_config['model_config'])
    else:
        model = AlgorithmRL.load(reload_model, env=env, device='cuda', **algo_config['model_config'])

    model.set_logger(new_logger)

    model.learn(total_timesteps=train_config['total_timesteps'],
                callback=[ TensorboardCallback(1), CheckpointCallback(
                    save_freq=train_config['total_timesteps'] // train_config["num_checkpoints"],
                    save_path=SAVE_PATH,
                    name_prefix="model")], reset_num_timesteps=False)
    
except Exception as e:
    print(e)

finally:
    env.close()