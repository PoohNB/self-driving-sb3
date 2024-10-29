from environment.loader import env_from_config
import traceback
from utils import TensorboardCallback,write_json,write_pickle,create_policy_paths,get_filtered_attribute_names
from config import observer_config,action_config

from config.algorithm_config import available_AlgorithmRL
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import os

from config.Config_loader import get_train_config
import argparse

def main():
    parser = argparse.ArgumentParser(description="Get configuration for the algorithm.")

    parser.add_argument('algo_config', type=str, help='name of config available in config/algorithm or algorithm configuration')
    parser.add_argument('--obs_module', choices=get_filtered_attribute_names(observer_config,"observer"),
                        default='observer_con_manv', 
                        help='name of observer available in config/observer_config (default: "observer_con_manv")')
    parser.add_argument('--act_wrapper', choices=get_filtered_attribute_names(action_config,"action"),
                        default='action_limit', 
                        help='name of observer available in config/action_config (default: "action_limit")')
    parser.add_argument('--discrete_actions', choices=get_filtered_attribute_names(action_config,"discrete"),
                        default=None, help='name of observer available in config/action_config (default: None)')
    parser.add_argument('--map_name', type=str, default='AIT', help='The name of the map (default: AIT)')
    parser.add_argument('--level', type=int, default=0, help='The level (default: 0)')    
    parser.add_argument('--total_timesteps', type=int, default=-1, help='total_timesteps')
    parser.add_argument('--max_step', type=int, default=-1, help='max_step')
    parser.add_argument('--load_model', type=str, default='', help='The path to the model to load (default: "")')
    parser.add_argument('--save_replay_buffer', action='store_true', help='save_replay_buffer for continue training (default: False)')

    parser.add_argument('--seed', type=int, default=2024, help='env seed (default: 2024)')
    parser.add_argument('--render', action='store_false', help='render image while training (default: True)')
    parser.add_argument('--log_dir', type=str, default="runs/RL", help='train log directory (default: "runs/RL")')
    parser.add_argument('--save_dir', type=str, default="RLmodel", help='model save directory (default: "RLmodel")')

    args = parser.parse_args()

    reload_model = args.load_model
    CONFIG = get_train_config(args.algo_config,
                              args.obs_module,
                              args.act_wrapper,
                              args.discrete_actions,
                              args.map_name,
                              args.level,
                              args.total_timesteps)
    LOG_DIR = args.log_dir
    SAVE_PATH = args.save_dir
    RENDER = args.render
    # =======================================================================
    algo_config = CONFIG["algorithm"]
    train_config = CONFIG['train_config']

    if algo_config['method'] not in available_AlgorithmRL:
        raise ValueError("Invalid algorithm name")

    SAVE_PATH,LOG_DIR = create_policy_paths(SAVE_PATH,LOG_DIR,algo_config['method'])

    CONFIG['env']['env_config']['seed'] = args.seed
    if args.max_step >0:
        print("set the max_step to ",args.max_step)
        CONFIG['env']['max_step'] = args.max_step

    env = env_from_config(CONFIG['env'],RENDER=RENDER)

    print("============ init algorithm.. ===============")
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
                        **algo_config['algo_config'])
        else:
            if algo_config['method'] =="SAC" or algo_config['method'] =="TQC":
                algo_config['algo_config']['policy_kwargs']['use_sde'] =True
            model = AlgorithmRL.load(reload_model,
                                      env=env,
                                       seed=algo_config['seed'],
                                        device='cuda', 
                                        **algo_config['algo_config'])
            
            replay_buffer_path = os.path.join(os.path.dirname(reload_model),"replay_buffer")
            if os.path.exists(replay_buffer_path):
                if hasattr(model, 'load_replay_buffer'):
                    model.load_replay_buffer(os.path.dirname(reload_model))
                    print("Replay buffer loaded.")
                else:
                    print("This algorithm does not use a replay buffer.")

        model.set_logger(new_logger)

        print("training.. ")
        model.learn(total_timesteps=train_config['total_timesteps'],
                    callback=[ TensorboardCallback(1), CheckpointCallback(
                        save_freq=train_config['total_timesteps'] // train_config["num_checkpoints"],
                        save_path=SAVE_PATH,
                        name_prefix="model")], reset_num_timesteps=False)
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"An error occurred: {e}")
        print(f"Traceback details:\n{tb}")

    finally:

        env.close()

        if hasattr(model, 'save_replay_buffer') and args.save_replay_buffer:
            model.save_replay_buffer(os.path.join(SAVE_PATH,"replay_buffer"))
            print("Replay buffer saved.")
        else:
            print("This algorithm does not use a replay buffer.")

if __name__ == "__main__":
    main()