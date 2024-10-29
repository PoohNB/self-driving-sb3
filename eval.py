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
    parser.add_argument('--save_score', action='store_true', help='save_score in json (default: False)')
    parser.add_argument('--save_info', action='store_true', help='save info in csv (default: False)')
    parser.add_argument('--tag', type=str, default='', help='video tag name (default: None)')
    parser.add_argument('--fps', type=int, default=-1, help='fps for record video (default: -1 (use env fps))')
    parser.add_argument('--speed_factor', type=float, default=1, help='speed factor for evaluate the effect of speed change to model (default:1)')
    parser.add_argument('--delta_frame', type=float, default=-1, help='delta frame between each action (default:-1 which mean use delta frame from config)')
    parser.add_argument('--max_step', type=int, default=1000, help='max step (default:1000)')

    args = parser.parse_args()

    # Paths and configurations=============
    model_path = args.load_model
    seed = args.seed
    record = args.record
    eval_times = args.eps
    set_delta_frame = args.delta_frame
    #=============================

    # load the saved env config for refference
    model_dir = os.path.dirname(model_path)
    checkpoint_name = os.path.basename(model_path)
    config_path = os.path.join(model_dir, "env_config.pkl")
    with open(config_path, 'rb') as file:
        loaded_env_config = pickle.load(file)

    # edit the env config 
    selected_level = get_level_config(args.map_name,args.level)
    selected_level['env']['max_step'] = args.max_step
    env_config1 = dict( **selected_level['env'],
                    discrete_actions = loaded_env_config['env_config']['discrete_actions']
                    )
    loaded_env_config['env_config'] = env_config1
    loaded_env_config['env_config']['seed']=seed
    loaded_env_config['env_config']['cam_config_list'][0]["Location"] = [0.98,0,1.675]
    loaded_env_config['env_config']['cam_config_list'][0]["Rotation"] = [-12.5,0,0]
    loaded_env_config['env_config']['cam_config_list'][0]["attribute"]["fov"] = 78
    
    if set_delta_frame >0:
        loaded_env_config['env_config']['carla_setting']['delta_frame'] = set_delta_frame
        print(f"set delta frame to {set_delta_frame}")

    # record save : video, info, result of test
    if record or args.save_score or args.save_info:
        record_path = os.path.join("recorded",model_dir.split('/')[-1])
        os.makedirs(record_path,exist_ok=True)
        video_name = "map"+args.map_name+str(args.level)+"_"+checkpoint_name.replace(".zip", ".avi")
        result_name = "map"+args.map_name+str(args.level)+"_"+"result.json"
        csv_name = "map"+args.map_name+str(args.level)+"_"+"infos.csv"
        if args.tag != "":
            video_name = args.tag+"_"+video_name
            result_name = args.tag+"_"+result_name
            csv_name = args.tag+"_"+csv_name
        video_path = os.path.join(record_path,video_name)
        result_path = os.path.join(record_path,result_name)
        csv_path = os.path.join(record_path,csv_name)
    # Load the configuration file
    # print(loaded_env_config)
    # set the selected seed
    loaded_env_config['seed'] = seed
    # add the vae decoder if vae encoder exist
    if "Vae" in loaded_env_config['observer_config']['name']:
        if loaded_env_config.get('observer_config', {}).get('config').get('vae_decoder_config') is None:
  
            vencoder_model_path = loaded_env_config['observer_config']['config']['vae_encoder_config']['model_path']
            vencoder_latent_dims = loaded_env_config['observer_config']['config']['vae_encoder_config']['latent_dims']
            decoder_model_path = os.path.join(os.path.dirname(vencoder_model_path), "decoder_model.pth")

            loaded_env_config['observer_config']['config']['vae_decoder_config'] = {
                'model_path': decoder_model_path,
                'latent_dims': vencoder_latent_dims
            }

    try:
        
        # Create environment
        env = env_from_config(loaded_env_config, True)
        env.eval_mode()
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
        ep_lens = []
        if record:
            if args.fps <0:
                fps = env.fps
            else:
                fps = args.fps
            print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *(720,1280,3),
                                                                int(fps)))
            video_recorder = VideoRecorder(video_path,
                                        frame_size=(720,1280,3),
                                        fps=fps)

        for episode in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            ep_len = 0
            rendered_frame = env.get_spectator_image()
            obs_shape = obs.shape[0]

            while not done:
        
                action, _ = model.predict(obs.reshape((1,obs_shape)) ,deterministic=True)#
                action = [action[0],max(min(action[1]*args.speed_factor,1),0)]
    
                obs, reward, done, info = env.step(action)
                total_reward += reward
                rendered_frame = env.get_spectator_image()
                ep_len+=1
                if record:
                    video_recorder.add_frame(rendered_frame)
                    data_list.append(info)


            rewards.append(total_reward)
            ep_lens.append(ep_len)

            logger.info(f"Episode {episode + 1}: Reward = {total_reward}, Length = {ep_len}")

        avg_reward = sum(rewards) / episodes
        avg_length = sum(ep_lens) / episodes
        

        logger.info(f"Average Reward over {episodes} episodes: {avg_reward}")
        logger.info(f"Average Length : {avg_length}")
        if args.save_score:
            result = dict(episodes=episodes,
                          rewards=rewards,
                          lengths=ep_lens,
                        avg_length=avg_length,
                        avg_reward=avg_reward
            )
            with open(result_path,'w') as f:
                json.dump(result,f, indent=4)
        if args.save_info:
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