from config.env import *

env_config_base = dict(cam_config_list=[front_cam], 
                    seed=2024,
                    rand_weather=False)

levels = [dict(env= dict(world_config = dict(**world_config_base,
                                       max_step =250),  
                        coach_config=dict(scene_configs=[scene_inner_easy,scene_outer_easy],
                                        **rest_config,)
                        ),
             total_timesteps = 200000
             ),
        dict(env= dict(world_config = dict(**world_config_base,
                                       max_step =1200),  
                        coach_config=dict(scene_configs=[scene_inner_med,scene_outer_med],
                                        **rest_config,)
                        ),
            total_timesteps = 100000
             ),
        dict(env= dict(world_config = dict(**world_config_base,
                        max_step =2000),  
                        coach_config=dict(scene_configs=[scene_inner_hard,scene_outer_hard],
                                        **rest_config,)
                        ),
            total_timesteps = 300000
            ),

        ] 
             