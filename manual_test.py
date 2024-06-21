from environment.loader import manualctrlenv_from_config

from config.trainRL_config import MANUAL_ENV
from environment.tools.env_wrapper import GymWrapper
import cv2
import time
import os,sys
import traceback

sync = True
env = manualctrlenv_from_config(MANUAL_ENV,sync)
try:

    obs = env.reset()
    done = False
    step = 0

    while not done:
        # action, _states = model.predict(obs.reshape((1,272)))#
        obs, reward, done, info = env.step()
        # if episode >0:
        #     time.sleep(1)

    print(info)

except Exception as e:
    # exc_type, exc_obj, exc_tb = sys.exc_info()
    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    # print(exc_type, fname, exc_tb.tb_lineno)
    # print(e)
    tb = traceback.format_exc()
    print(f"An error occurred: {e}")
    print(f"Traceback details:\n{tb}")

finally:
    env.close()