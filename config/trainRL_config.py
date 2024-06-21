from config.env import *
from config.algorithm_config import *
from config.observer_config import *
from config.seg_config import *
from config.vae import *

# action ===============================================
limitaction1 = dict(name="LimitAction",
                    config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.8,
                                                    steer_still_range = 0.1))

limitaction2 = dict(name="LimitAction",
                    config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.8,
                                                    steer_still_range = 0))


originaction = dict(name="OriginAction",
                    config=dict(steer_still_range = 0))

# observer ============================================

observer1 = dict(name="SegVaeActHistObserver",
                 config=dict(num_img_input = 1,
                            act_num=2,
                            hist_len = 8,
                            skip_frame=0))

observer_no_hist = dict(name="SegVaeActObserver",
                 config=dict(num_img_input = 1,
                            act_num=2,))

observer_discrete = dict(name="SegVaeActHistObserver",
                 config=dict(num_img_input = 1,
                            act_num=1,
                            hist_len = 8,
                            skip_frame=0))


# env_config ===============================================================

env_config1 = dict(env_setting = dict(**env_config_base,max_step =500),
                    cam_config_list=[front_cam], 
                    discrete_actions = None,
                    coach_config = ait_fb_scene_outer_only,
                    seed=2024,
                    rand_weather=False)
  
manual_env_config = dict(env_setting = dict(**env_config_base,max_step =1000),
                        cam_config_list=[front_cam], 
                        discrete_actions = [[-0.6,0.4],[-0.1,0.56],[0,0.6],[0,0.4],[0,0],[0.1,0.56],[0.6,0.4]],
                        coach_config = ait_fb_scene_outer_only,
                        seed=2024,
                        rand_weather=False)

# [[0.0,0],[0.1,0],[0.2,0],[0.3,0],[0.4,0],[0.5,0],[0.6,0],[0.7,0],[0.8,0]],
# [-0.6,0.4],[-0.1,0.56],[0,0.6],[0,0.4],[0,0],[0.1,0.56],[0.6,0.4]
# all_config =======================================

test_ENV = dict(observer_config=observer1,
            seg_config=fbm2f_fp16_1280,
            vencoder_config=vencoder32,
            decoder_config=decoder32,
            env_config=env_config1,
            actionwrapper= originaction)

MANUAL_ENV = dict(observer_config=observer_discrete,
            seg_config=fbm2f_fp16_1280,
            vencoder_config=vencoder32,
            decoder_config=decoder32,
            env_config=manual_env_config,
            actionwrapper= originaction)

ENV1 = dict(observer_config=observer1,
            seg_config=fbm2f_fp16,
            vencoder_config=vencoder32,
            decoder_config=decoder32,
            env_config=env_config1,
            actionwrapper= limitaction1,)

ENV3 = dict(observer_config=observer1,
            seg_config=fbm2f_fp16,
            vencoder_config=vencoder32,
            decoder_config=None,
            env_config=env_config1,
            actionwrapper= limitaction2,)

ENV4RNN = dict(observer_config=observer_no_hist,
            seg_config=fbm2f_fp16,
            vencoder_config=vencoder32,
            decoder_config=None,
            env_config=env_config1,
            actionwrapper= limitaction2)



# train experiment config =======================================
RL1 = dict(env=ENV1,
           train=dict(
            algorithm=dict(method="SAC",model_config=SAC1,seed=2024),
            train_config=dict(total_timesteps=100000,
                              num_checkpoints=5,
                              save_path="RLmodel")))


RL_SAC_v2 = dict(env=ENV3,
           train=dict(
            algorithm=dict(method="SAC",model_config=SAC1,seed=2024),
            train_config=dict(total_timesteps=100000,
                              num_checkpoints=5,
                              save_path="RLmodel")))
RL_SAC_v2_con = dict(env=ENV3,
           train=dict(
            algorithm=dict(method="SAC",model_config=SAC1_con,seed=2024),
            train_config=dict(total_timesteps=100000,
                              num_checkpoints=5,
                              save_path="RLmodel")))

RL_test2 = dict(env=ENV3,
           train=dict(
            algorithm=dict(method="PPO",model_config=PPO1,seed=2024),
            train_config=dict(total_timesteps=100000,
                              num_checkpoints=5,
                              save_path="RLmodel")))
RL_test3 = dict(env=ENV3,
           train=dict(
            algorithm=dict(method="PPO",model_config=PPO2,seed=2024),
            train_config=dict(total_timesteps=100000,
                              num_checkpoints=5,
                              save_path="RLmodel")))

RL_rnnppo_1 = dict(env=ENV4RNN,
           train=dict(
            algorithm=dict(method="RecurrentPPO",model_config=RNNPPO1,seed=2024),
            train_config=dict(total_timesteps=70000,
                              num_checkpoints=5,
                              save_path="RLmodel")))



