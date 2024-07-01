from config.levels import levels,env_config_base
from config.algorithm_config import SAC1
from config.observer_config import *
from config.vae import *
from config.seg_config import *
from config.env.action_config import limitaction2


select_level = levels[2]

env_config_base['discrete_actions'] = None

env_config1 = dict( **select_level['env'],
                    **env_config_base
                     )


ENV_SAC = dict(observer_config=observer_con_manv,
            seg_config=fbm2f_fp16_1280,
            vencoder_config=vencoder32,
            decoder_config=None,
            env_config=env_config1,
            actionwrapper= limitaction2,)

RL_SAC1 = dict(env=ENV_SAC,
           train=dict(
            algorithm=dict(method="SAC",model_config=SAC1,seed=2024),
            train_config=dict(total_timesteps=select_level['total_timesteps'],
                              num_checkpoints=5,
                              save_path="RLmodel")))