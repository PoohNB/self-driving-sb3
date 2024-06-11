from config.env import *
from config.algorithm_config import *
from config.observer_config import *
from config.seg_config import *
from config.vae import *

limitaction1 = dict(name="LimitAction",
                    action_config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.8,
                                                    steer_still_range = 0.1))

limitaction2 = dict(name="LimitAction",
                    action_config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.8,
                                                    steer_still_range = 0))


originaction = dict(name="OriginAction")

observer1 = dict(latent_space=32,
                            num_img_input = 1,
                            act_num=2,
                            hist_len = 8,
                            skip_frame=0)

env_config1 = dict(car_spawn=ait_football_spawn,
                            spawn_mode='static',
                            env_config = dict(**env_config_base,max_step =500),
                            cam_config_list=[front_cam], 
                            discrete_actions = None,
                            seed=2024,
                            rand_weather=False)

env_config2 = dict(car_spawn=ait_football_spawn,
                            spawn_mode='random',
                            env_config = dict(**env_config_base,max_step =500),
                            cam_config_list=[front_cam], 
                            discrete_actions = None,
                            seed=2024,
                            rand_weather=False)

env_config3 = dict(car_spawn=ait_football_spawn,
                            spawn_mode='random',
                            env_config = dict(**env_config_base,max_step =500),
                            cam_config_list=[front_cam], 
                            discrete_actions = None,
                            seed=2024,
                            rand_weather=True)


# environment config =======================================
ENV1 = dict(observer_config=observer1,
            seg_config=fbm2f_fp16,
            vencoder_config=vencoder32,
            decoder_config=decoder32,
            env_config=env_config1,
            actionwrapper= limitaction1,
            rewarder_config = dict(mask_path = "environment/rewardmask/ait_map/ait_football.png"),)

ENV2 = dict(observer_config=observer1,
            seg_config=fbm2f_fp16,
            vencoder_config=vencoder32,
            decoder_config=None,
            env_config=env_config1,
            actionwrapper= limitaction2,
            rewarder_config = dict(mask_path = "environment/rewardmask/ait_map/ait_football_thick.png"))


# train experiment config =======================================
RL1 = dict(env=ENV1,
           train=dict(
            algorithm=dict(policy="SAC",model_config=SAC1,seed=2024),
            train_config=dict(total_timesteps=100000,
                              num_checkpoints=5,
                              save_path="RLmodel")))

RL_test = dict(env=ENV2,
           train=dict(
            algorithm=dict(policy="SAC",model_config=SAC1,seed=2024),
            train_config=dict(total_timesteps=100000,
                              num_checkpoints=5,
                              save_path="RLmodel")))

