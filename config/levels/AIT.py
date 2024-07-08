from config.env import *

env_config_base = dict(cam_config_list=[front_cam], 
                    seed=2024,
                    rand_weather=False)

levels = [dict(env= dict(world_config = dict(**world_config_base,
                                       max_step =250),  
                        coach_config=dict(scene_configs=[AIT_scenes.scene_inner_easy,AIT_scenes.scene_outer_easy],
                                        **AIT_scenes.coach_base,)
                        ),
             total_timesteps = 200000
             ),
        dict(env= dict(world_config = dict(**world_config_base,
                                       max_step =1200),  
                        coach_config=dict(scene_configs=[AIT_scenes.scene_inner_med,AIT_scenes.scene_outer_med],
                                        **AIT_scenes.coach_base,)
                        ),
            total_timesteps = 100000
             ),
        dict(env= dict(world_config = dict(**world_config_base,
                        max_step =2000),  
                        coach_config=dict(scene_configs=[AIT_scenes.scene_inner_hard,AIT_scenes.scene_outer_hard],
                                        **AIT_scenes.coach_base,)
                        ),
            total_timesteps = 1000000
            ),

        ] 
             