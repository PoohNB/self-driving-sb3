from environment.loader import env_from_config

from config.trainRL_config import RL_test

CONFIG = RL_test
episodes = 2
action = [0.,1.0]
env = env_from_config(CONFIG['env'],True)
try:


    for episode in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            # action, _states = model.predict(obs.reshape((1,272)))#
            obs, reward, done, info = env.step(action)

except Exception as e:
    print(e)

finally:
    env.close()