from env import *
from algorithm_config import *
from observer_config import *
from seg_config import *

RL1 = dict(observer_config=dict(latent_space=32,
                            num_img_input = 1,
                            act_num=2,
                            hist_len = 8,
                            skip_frame=0),
            seg_config=dict(model_repo="facebook/mask2former-swin-large-mapillary-vistas-semantic",
                            fp16=True,
                            labelmap=mask2former_labelmap,
                            crop=(512,1024)),
            vae_config=dict(model_path="autoencoder/model/vae32/best/var_encoder_model.pth",
                            latent_dims=32),
            decoder_config=dict(model_path="autoencoder/model/vae32/best/decoder_model.pth",
                            latent_dims=32),
            env_config=dict(car_spawn=ait_football_spawn,
                            spawn_mode='static',
                            env_config = dict(**env_config_base,max_step =1000),
                            cam_config_list=[front_cam], 
                            discrete_actions = None,
                            seed=2024,
                            rand_weather=False),
            actionwrapper=dict(name="LimitAction",config=dict(**limit_action1,stable_steer=True)),
            rewarder_config = dict(mask_path = "environment/rewardmask/ait_map/ait_football.png"),
            algorithm=dict(policy="SAC",model_config=SAC1,seed=2024),
            train_config=dict(total_timesteps=50000,
                              num_checkpoints=10,
                              save_path="RLmodel"))

RL2 = dict()

