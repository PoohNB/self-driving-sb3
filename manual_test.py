from environment.loader import manualctrlenv_from_config

from config.trainRL_config import MANUAL_ENV
from environment.tools.env_wrapper import GymWrapper
import cv2
import time
sync = True
env = manualctrlenv_from_config(MANUAL_ENV,sync)
try:


    obs = env.reset()
    done = False
    step = 0

    while not done:
        # action, _states = model.predict(obs.reshape((1,272)))#
        obs, reward, done, info = env.step()
        step+=1
        time.sleep(1)
        # if episode >0:
        #     time.sleep(1)


except Exception as e:
    print(e)

finally:
    env.close()