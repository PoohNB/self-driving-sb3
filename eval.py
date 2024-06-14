import pickle
import os
import json
import pandas as pd

import logging
from stable_baselines3 import SAC, PPO, DDPG
from sb3_contrib import RecurrentPPO
from environment.loader import env_from_config
from utils import VideoRecorder
from config.trainRL_config import ENV2
from environment.tools.env_wrapper import GymWrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available policies
available_policy = {
    "SAC": SAC,
    "PPO": PPO,
    "DDPG": DDPG,
    "RecurrentPPO": RecurrentPPO
}

# Paths and configurations=============
model_path = "RLmodel/SAC_5/model_100000_steps.zip"
# model_path = "optuna_trials/PPO/trial_21/best_model.zip"
manual_config = None
seed = 2024
record = True # get 1 video and 1 csv file of info of each step
eval_times = 2
#=============================

if manual_config is None:
    model_dir = os.path.dirname(model_path)
    checkpoint_name = os.path.basename(model_path)
    config_path = os.path.join(model_dir, "env_config.pkl")



if record:
    record_path = os.path.join("recorded",model_dir.split('/')[-1])
    os.makedirs(record_path,exist_ok=True)
    video_path = os.path.join(record_path,checkpoint_name.replace(".zip", "_eval.avi"))
    result_path = os.path.join(record_path,"result.json")
    csv_path = os.path.join(record_path,"infos.csv")
# Load the configuration file


if manual_config is not None:
    env_config = manual_config

else:
    with open(config_path, 'rb') as file:
        env_config = pickle.load(file)

env_config['seed'] = seed
if env_config['decoder_config'] is None:
    env_config['decoder_config'] = dict(model_path =os.path.join(os.path.dirname(env_config['vencoder_config']['model_path']),"decoder_model.pth"),
                                        latent_dims=env_config['vencoder_config']['latent_dims'])

# CONFIG = RL1

try:
    
    # Create environment
    env = env_from_config(env_config, True)
    env = GymWrapper(env)

    # Load the model
    Policy = available_policy[model_path.split('/')[1].split('_')[0]]
    model = Policy.load(model_path, env=env, device='cuda')

    logger.info("Model and environment loaded successfully.")

    # Evaluate the model
    episodes = eval_times
    rewards = []
    data_list = []

    if record:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *(720,1280,3),
                                                            int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                    frame_size=(720,1280,3),
                                    fps=env.fps)

    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        rendered_frame = env.get_spectator_image()
        obs_shape = obs.shape[0]

        while not done:
            action, _states = model.predict(obs.reshape((1,obs_shape)))#
            obs, reward, done, info = env.step(action)
            total_reward += reward
            rendered_frame = env.get_spectator_image()
            if record:
                video_recorder.add_frame(rendered_frame)
                data_list.append(info)


        rewards.append(total_reward)
        logger.info(f"Episode {episode + 1}: Reward = {total_reward}")

    avg_reward = sum(rewards) / episodes
    

    logger.info(f"Average Reward over {episodes} episodes: {avg_reward}")

    if record:
        result = dict(episodes=episodes,
                    avg_reward=avg_reward
        )
        with open(result_path,'w') as f:
            json.dump(result,f, indent=4)

        df = pd.DataFrame(data_list)
        df.to_csv(csv_path, index=False)

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")

finally:
    if record:
        video_recorder.release()

    env.close()
