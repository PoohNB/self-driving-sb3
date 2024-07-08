from config.observer_config import *
from config.vae import *
from config.seg_config import *
from config.env.action_config import *
from config import algorithm_config
import ast

from config import levels

def get_env_config(action_type,map_name="AIT",level=0):
    selected_map = getattr(levels,map_name)
    selected_level = selected_map.levels[level]
    env_config_base = selected_map.env_config_base

    env_config_base['discrete_actions'] = None

    env_config1 = dict( **selected_level['env'],
                        **env_config_base
                        )

    env_config_base['discrete_actions'] = discret_actions

    env_config2 = dict( **selected_level['env'],
                        **env_config_base
                        )

    if action_type == "continuous":

        envEx= dict(observer_config=observer_con_manv,
                seg_config=fbm2f_fp16_1280,
                vencoder_config=vencoder32,
                decoder_config=None,
                env_config=env_config1,
                actionwrapper= limitaction2,)
    
    elif action_type == "discrete":

        envEx= dict(observer_config=observer_con_manv,
                seg_config=fbm2f_fp16_1280,
                vencoder_config=vencoder32,
                decoder_config=None,
                env_config=env_config2,
                actionwrapper= None)
    
    else:
        raise Exception(" action_type can be continuous or discrete")
    
    return envEx,selected_level


def get_config(algorithm,model_config,action_type,map_name="AIT",level=0):

    envEx,selected_level = get_env_config(action_type=action_type,
                   map_name=map_name,
                   level=level)

    try:
        model_config = ast.literal_eval(model_config)
        if not isinstance(model_config, dict):
            raise ValueError
    except (ValueError, SyntaxError):
        model_config = getattr(algorithm_config,model_config)

    num_checkpoints = 5
    if selected_level['total_timesteps'] > 200000:
        num_checkpoints = 10
        print(f"total_timesteps:{selected_level['total_timesteps']} update num_checkpoints to {num_checkpoints}")


    return dict(env=envEx,
           train=dict(
            algorithm=dict(method=algorithm,model_config=model_config,seed=2024),
            train_config=dict(total_timesteps=selected_level['total_timesteps'],
                              num_checkpoints=num_checkpoints,
                              save_path="RLmodel")))