# Test the environment

from environment.ENV_MANUAL_ROUTE import CarlaEnv

if __name__ == "__main__":
    env = CarlaEnv()
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)