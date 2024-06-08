import json
import os
import logging
from stable_baselines3 import SAC, PPO, DDPG
from sb3_contrib import RecurrentPPO
from environment.loader import env_from_config
from utils import VideoRecorder
from config.trainRL_config import RL1

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

# Paths and configurations
model_path = "RLmodel/SAC_6_testrun/model_100000_steps.zip"
dir = os.path.dirname(model_path)
config_path = os.path.join(dir, "config.json")
seed = 2024
record = False
video_path = os.path.join("recorded",dir.split('/')[-1])
os.makedirs(video_path,exist_ok=True)
# Load the configuration file
# with open(config_path, 'r') as file:
#     CONFIG = json.load(file)

# CONFIG['env_config']['seed'] = seed

CONFIG = RL1

try:

    # Create environment
    env = env_from_config(CONFIG, True)

    # Load the model
    Policy = available_policy[CONFIG['algorithm']['policy']]
    model = Policy.load(model_path, env=env, device='cuda')

    logger.info("Model and environment loaded successfully.")

    # Evaluate the model
    episodes = 2
    rewards = []

    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        # if record:
        #     print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
        #                                                         int(env.fps)))
        #     video_recorder = VideoRecorder(video_path,
        #                                 frame_size=rendered_frame.shape,
        #                                 fps=env.fps)

        while not done:
            action, _states = model.predict(obs.reshape((1,272)))#
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # rendered_frame = env.render(mode='rgb_array')
            # if record:
            #     video_recorder.add_frame(rendered_frame)


        rewards.append(total_reward)
        logger.info(f"Episode {episode + 1}: Reward = {total_reward}")

    avg_reward = sum(rewards) / episodes
    logger.info(f"Average Reward over {episodes} episodes: {avg_reward}")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")

finally:
    if record:
        video_recorder.release()

    env.close()
