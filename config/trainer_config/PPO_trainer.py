from config.levels import levels,env_config_base
from config.algorithm_config import PPO1
from config.observer_config import *
from config.vae import *
from config.seg_config import *
from config.env.action_config import limitaction2

             
select_level = levels[0]

env_config_base['discrete_actions'] = None

env_config1 = dict( **select_level['env'],
                    **env_config_base
                     )


ENV_PPO = dict(observer_config=observer_con_manv,
            seg_config=fbm2f_fp16_1280,
            vencoder_config=vencoder32,
            decoder_config=None,
            env_config=env_config1,
            actionwrapper= limitaction2,)

RL_PPO1 = dict(env=ENV_PPO,
           train=dict(
            algorithm=dict(method="PPO",model_config=PPO1,seed=2024),
            train_config=dict(total_timesteps=select_level['total_timesteps'],
                              num_checkpoints=5,
                              save_path="RLmodel")))