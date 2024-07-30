import pickle
import os
import json
import pandas as pd
import argparse
import traceback

import logging
from environment.loader import env_from_config
from config.Config_loader import get_level_config
from utils import VideoRecorder
from environment.tools.env_wrapper import GymWrapper
from config.algorithm_config import available_AlgorithmRL


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(description="Get configuration for env testing")
    parser.add_argument('load_model', type=str, help='Path to agent model')
    parser.add_argument('--map_name', type=str, default='AIT', help='The name of the map (default: AIT)')
    parser.add_argument('--level', type=int, default=0, help='The level (default: 0)')
    parser.add_argument('--eps', type=int, default=1, help='test episode (default:1)')
    parser.add_argument('--seed', type=int, default=2077, help='env seed (default: 2077)')
    parser.add_argument('--record', action='store_true', help='record the video in recorded folder (default: False)')

    args = parser.parse_args()

    # Paths and configurations=============
    model_path = args.load_model
    seed = args.seed
    record = args.record
    eval_times = args.eps
    #=============================

    # load the saved env config for refference
    model_dir = os.path.dirname(model_path)
    checkpoint_name = os.path.basename(model_path)
    config_path = os.path.join(model_dir, "env_config.pkl")
    with open(config_path, 'rb') as file:
        loaded_env_config = pickle.load(file)

    # edit the env config 
    selected_level = get_level_config(args.map_name,args.level)
    env_config1 = dict( **selected_level['env'],
                    discrete_actions = loaded_env_config['env_config']['discrete_actions']
                    )
    loaded_env_config['env_config'] = env_config1
    loaded_env_config['env_config']['seed']=seed

    # record save : video, info, result of test
    if record:
        record_path = os.path.join("recorded",model_dir.split('/')[-1])
        os.makedirs(record_path,exist_ok=True)
        video_path = os.path.join(record_path,checkpoint_name.replace(".zip", "_eval.avi"))
        result_path = os.path.join(record_path,"result.json")
        csv_path = os.path.join(record_path,"infos.csv")
    # Load the configuration file
    # print(loaded_env_config)
    # set the selected seed
    loaded_env_config['seed'] = seed
    # add the vae decoder if vae encoder exist
    if "Vae" in loaded_env_config['observer_config']['name']:
        if loaded_env_config.get('observer_config', {}).get('vae_decoder_config') is None:
  
            vencoder_model_path = loaded_env_config['observer_config']['vae_encoder_config']['model_path']
            vencoder_latent_dims = loaded_env_config['observer_config']['vae_encoder_config']['latent_dims']
            decoder_model_path = os.path.join(os.path.dirname(vencoder_model_path), "decoder_model.pth")

            loaded_env_config['observer_config']['vae_decoder_config'] = {
                'model_path': decoder_model_path,
                'latent_dims': vencoder_latent_dims
            }

    try:
        
        # Create environment
        env = env_from_config(loaded_env_config, True)
        env = GymWrapper(env)

        # Load the model
        policy_name = model_path.split('/')[1].split('_')[0]
        Policy = available_AlgorithmRL[policy_name]
        model = Policy.load(model_path, device='cuda')

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
        
                action, _ = model.predict(obs.reshape((1,obs_shape)) ,deterministic=True)#
    
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
        tb = traceback.format_exc()
        print(f"An error occurred: {e}")
        print(f"Traceback details:\n{tb}")

    finally:
        if record:
            video_recorder.release()

        env.close()


if __name__ == "__main__":
    main()