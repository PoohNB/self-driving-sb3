from config.levels import levels,env_config_base
from config.algorithm_config import DQN1
from config.observer_config import *
from config.vae import *
from config.seg_config import *
from config.env.action_config import discret_actions

             
select_level = levels[0]

env_config_base['discrete_actions'] = discret_actions

env_config1 = dict( **select_level['env'],
                    **env_config_base
                     )


ENV_DQN = dict(observer_config=observer_con_manv,
            seg_config=fbm2f_fp16_1280,
            vencoder_config=vencoder32,
            decoder_config=None,
            env_config=env_config1,
            actionwrapper= None)

RL_DQN1 = dict(env=ENV_DQN,
           train=dict(
            algorithm=dict(method="PPO",model_config=DQN1,seed=2024),
            train_config=dict(total_timesteps=select_level['total_timesteps'],
                              num_checkpoints=5,
                              save_path="RLmodel")))