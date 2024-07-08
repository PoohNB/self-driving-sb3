from environment.loader import env_from_config
from environment.tools.env_wrapper import GymWrapper
import time
from config.Config_loader import get_env_config
import argparse
import traceback

def env_run(args):

    episodes = 1
    env_config = get_env_config()
    env_config['env_config']['seed']=args.seed
    env = env_from_config(env_config,True)
    # env = GymWrapper(env)
    try:


        for episode in range(episodes):
            obs,_ = env.reset()
            done = False
            trunt = False
            step = 0

            while not (trunt or done):
                if args.mode == "manual":
                    action= env.parse_control()
                else:
                    if step < 40:
                        action = [0.0,1.0]
                    else:
                        action = [1.0,1.0]
                obs, reward, done,trunt, info = env.step(action)
                step+=1
                # time.sleep(0.5)
                # if episode >0:
                #     time.sleep(1)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"An error occurred: {e}")
        print(f"Traceback details:\n{tb}")
    finally:
        env.close()


def main():

    parser = argparse.ArgumentParser(description="Get configuration for the algorithm.")
    parser.add_argument('--map_name', type=str, default='AIT', help='The name of the map (default: AIT)')
    parser.add_argument('--level', type=int, default=0, help='The level (default: 0)')
    parser.add_argument('--eps', type=int, default=1, help='test episode (default:1)')
    parser.add_argument('--seed', type=int, default=2077, help='env seed (default: 2077)')
    parser.add_argument('--mode', type=int, default="manual", help='mode "manual" or "test"')

    args = parser.parse_args()

    env_run(args)


if __name__ == "__main__":
    main()
