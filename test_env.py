from environment.loader import env_from_config

from config.trainRL_config import RL1
from environment.tools.env_wrapper import GymWrapper
import time
CONFIG = RL1
episodes = 2
action = [0.0,0.6]
CONFIG['env']['env_config']['seed'] =1231

env = env_from_config(CONFIG['env'],True)
env = GymWrapper(env)
try:


    for episode in range(episodes):
        obs = env.reset()
        done = False
        step = 0

        while not done:
            # action, _states = model.predict(obs.reshape((1,272)))#
            # if step > 50:
            #     action = [0.8,0.4]
            obs, reward, done, info = env.step(action)
            step+=1
            # if episode >0:
            #     time.sleep(1)

except Exception as e:
    print(e)

finally:
    env.close()