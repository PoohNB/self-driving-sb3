from config import observer_config
from config import action_config
from config import algorithm_config
import ast
import logging
import re

from config import levels

"""
get_level_config:
    args:
        map_name: map name in config/levels
        level: prepared level in that map

    return:
        selected_level(dict)

get_env_config:
    args:
        obs_module: for preprocess the observation, available in config/observer_config
        act_wrapper: for post process the action, available in config/action_config
        map_name: map name in config/levels
        level: prepared level in that map
    
    return:
        object_config(dict):{
            observer_config,
            env_config,
            action wrapper_config}
        selected_level(dict)


get_train_config:
    args:
        algorithm: name of algorithm
        algo_config: prepared algorithm config in config/algorithm

    return:
        env_config: full env config
        algorithm: algorithm config
        train_config: total_numsteps, num_checkpoints, save_path



"""

def extract_name(s):
    match = re.match(r"([a-zA-Z]+)", s)
    if match:
        return match.group(1)
    return None

def get_level_config(map_name="AIT",level=0):
    selected_map = getattr(levels,map_name)
    try:
        selected_level = selected_map.levels[level]
    except:
        logging.info("the level exceed highest level, set level to highest level")
        selected_level = selected_map.levels[-1]

    return selected_level


def get_env_config(obs_module="observer_con_manv",
                   act_wrapper="action_limit",
                   discrete_actions=None,
                   map_name="AIT",
                   level=0):

    obs_config = getattr(observer_config,obs_module)

    act_wrap = None if discrete_actions is not None else getattr(action_config, act_wrapper)

    selected_level=get_level_config(map_name,level)

    envEx= dict(observer_config=obs_config,
            env_config=dict( **selected_level['env'],
                discrete_actions = discrete_actions
                ),
            actionwrapper= act_wrap)
    
    return envEx,selected_level


def get_train_config(algo_config,
                     obs_module,
                     act_wrapper,
                     discrete_actions=None,
                     map_name="AIT",
                     level=0,
                     user_total_timesteps = -1):

    envEx,selected_level = get_env_config(obs_module=obs_module,
                                        act_wrapper=act_wrapper,
                                        discrete_actions=discrete_actions,
                                        map_name=map_name,
                                        level=level)
    algorithm=extract_name(algo_config)

    try:
        algo_config = ast.literal_eval(algo_config)
        if not isinstance(algo_config, dict):
            raise ValueError
    except (ValueError, SyntaxError):
        algo_config = getattr(algorithm_config,algo_config)

    if user_total_timesteps < 0:
        tts = selected_level['total_timesteps']
    else:
        tts = user_total_timesteps

    # determine num checkpoints
    if tts >= 400000:
        save_steps = 100000
        num_checkpoints = int(tts/save_steps)
    elif tts >= 200000:
        save_steps = 50000
        num_checkpoints = int(tts/save_steps)
    else: 
        save_steps = 25000
        num_checkpoints = int(tts/save_steps)

    print(f"num_checkpoints:{num_checkpoints}, save every {save_steps} timesteps")


    return dict(env=envEx,
                algorithm=dict(method=algorithm,algo_config=algo_config,seed=2024),
                train_config=dict(total_timesteps=tts,
                              num_checkpoints=num_checkpoints,
                              save_path="RLmodel"))