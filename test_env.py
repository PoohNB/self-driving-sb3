from environment.loader import env_from_config

from config.trainer_config import ENV_SAC
from environment.tools.env_wrapper import GymWrapper
import time
episodes = 5
action = [0.0,0.7]
ENV_SAC['env_config']['seed']=2025

env = env_from_config(ENV_SAC,True)
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