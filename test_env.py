from environment.loader import env_from_config
from environment.tools.env_wrapper import GymWrapper
import time
from config.Config_loader import get_env_config
import argparse
import traceback
import ast
from utils import get_filtered_attribute_names
from config import observer_config,action_config

def env_run(args):

    episodes = args.eps
    ENV_config,_ = get_env_config(obs_module=args.obs_module,
                   act_wrapper=args.act_wrapper,
                   discrete_actions=None,
                   map_name=args.map_name,
                   level=args.level)

    ENV_config['env_config']['seed']=args.seed
    if args.mode == "manual":
        ENV_config['actionwrapper'] = dict(name="ActionBase",
                    config=dict())
    env = env_from_config(ENV_config,True)
    if args.mode == "manual":
        env.pygamectrl.init_control(args.control)
    # env = GymWrapper(env)
    try:


        for _ in range(episodes):
            _,_ = env.reset()
            trunt = False
            step = 0

            while not trunt:
                if args.mode == "manual":
                    action= env.pygamectrl.action
                    # if step%5==0:
                    #     print(action)
                if args.mode == "auto":
                    action = args.control[2]
                _, _, done,trunt, info = env.step(action)
                step+=1

                if args.mode == "auto":
                    if done:
                        break
                if args.delay:
                    time.sleep(1)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"An error occurred: {e}")
        print(f"Traceback details:\n{tb}")
    finally:
        env.close()


def main():
    def parse_control(arg):
        return ast.literal_eval(arg)

    parser = argparse.ArgumentParser(description="Get configuration for env testing")
    parser.add_argument('--control', type=parse_control, default=[[-0.6,0.4],[-0.1,0.56],[0,0.6],[0,0.4],[0,0],[0.1,0.56],[0.6,0.4]], help='control list for number 1-10 in keyboard')
    parser.add_argument('--obs_module', choices=get_filtered_attribute_names(observer_config,"observer"),
                        default='observer_con_manv', 
                        help='name of observer available in config/observer_config (default: "observer_con_manv")')
    parser.add_argument('--act_wrapper', choices=get_filtered_attribute_names(action_config,"action"),
                        default='action_limit', 
                        help='name of observer available in config/action_config (default: "action_limit")')
    parser.add_argument('--map_name', type=str, default='AIT', help='The name of the map (default: AIT)')
    parser.add_argument('--level', type=int, default=0, help='The level (default: 0)')    
    parser.add_argument('--total_timesteps', type=int, default=-1, help='total_timesteps')
    parser.add_argument('--eps', type=int, default=1, help='test episode (default:1)')
    parser.add_argument('--seed', type=int, default=2077, help='env seed (default: 2077)')
    parser.add_argument('--mode', choices=['auto', 'manual'], default='manual', help='Mode of operation: auto or manual (default: manual)')
    parser.add_argument('--delay',action='store_true',help='delay step for see the change in slow motion' )

    args = parser.parse_args()

    env_run(args)


if __name__ == "__main__":
    main()
