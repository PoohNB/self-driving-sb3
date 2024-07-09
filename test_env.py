from environment.loader import env_from_config
from environment.tools.env_wrapper import GymWrapper
import time
from config.Config_loader import get_env_config
import argparse
import traceback

def env_run(args):

    episodes = 1
    env_config,_ = get_env_config("continuous",args.map_name,args.level)
    env_config['env_config']['seed']=args.seed
    if args.mode == "manual":
        env_config['actionwrapper'] = dict(name="OriginAction",
                    config=dict(steer_still_range = 0))
    env = env_from_config(env_config,True)
    if args.mode == "manual":
        env.pygamectrl.init_control()
    # env = GymWrapper(env)
    try:


        for episode in range(episodes):
            obs,_ = env.reset()
            trunt = False
            step = 0

            while not trunt:
                if args.mode == "manual":
                    action= env.pygamectrl.action
                    # if step%5==0:
                    #     print(action)
                if args.mode == "auto":
                    if step < 40:
                        action = [0.0,1.0]
                    else:
                        action = [1.0,1.0]
                obs, reward, done,trunt, info = env.step(action)
                step+=1

                if args.mode == "auto":
                    if done:
                        break
                # time.sleep(0.5)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"An error occurred: {e}")
        print(f"Traceback details:\n{tb}")
    finally:
        env.close()


def main():

    parser = argparse.ArgumentParser(description="Get configuration for env testing")
    parser.add_argument('--map_name', type=str, default='AIT', help='The name of the map (default: AIT)')
    parser.add_argument('--level', type=int, default=0, help='The level (default: 0)')
    parser.add_argument('--eps', type=int, default=1, help='test episode (default:1)')
    parser.add_argument('--seed', type=int, default=2077, help='env seed (default: 2077)')
    parser.add_argument('--mode', choices=['auto', 'manual'], default='manual', help='Mode of operation: auto or manual (default: manual)')

    args = parser.parse_args()

    env_run(args)


if __name__ == "__main__":
    main()
