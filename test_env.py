from environment.loader import env_from_config

from config.trainRL_config import test_ENV
from environment.tools.env_wrapper import GymWrapper
import time
episodes = 1
action = [0.0006,0.6]
test_ENV['env_config']['seed'] =1231

env = env_from_config(test_ENV,True)
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