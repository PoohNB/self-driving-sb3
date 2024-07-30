from config.env import *
from config.env.coach_config import AIT_scenes


"""
config of Scenes and timestep

"""

env_config_base = dict(cam_config_list=[front_cam], 
                    seed=2024
                    )
env_config_aug = dict(**env_config_base,
                      augment_image = True)

levels = [dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=250,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.curve_admin2aic_1),
                                                         get_scene_config(AIT_scenes.curve_admin2aic_2),
                                                         get_scene_config(AIT_scenes.curve_aic2admin_1),
                                                         get_scene_config(AIT_scenes.curve_aic2admin_2),
                                                         get_scene_config(AIT_scenes.ait_football_inner_straight_only),
                                                         get_scene_config(AIT_scenes.ait_football_outer_straight_only)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 50000
             ),
        dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=1800,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.ait_football_inner),
                                                         get_scene_config(AIT_scenes.ait_football_outer),],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 50000
             ),
        dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=250,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.curve_admin2aic_1),
                                                         get_scene_config(AIT_scenes.curve_admin2aic_2),
                                                         get_scene_config(AIT_scenes.curve_aic2admin_1),
                                                         get_scene_config(AIT_scenes.curve_aic2admin_2),
                                                         get_scene_config(AIT_scenes.ait_football_inner),
                                                         get_scene_config(AIT_scenes.ait_football_outer),
                                                         get_scene_config(AIT_scenes.turn_aic2admin),
                                                         get_scene_config(AIT_scenes.turn_admin2aic)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000
             ),

        dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=2500,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.aic2admin)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000
             ),

        dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=2500,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.admin2aic)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000
             ),
        dict(env= dict(**env_config_aug,
                         carla_setting = carla_setting,
                         max_step=2500,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.aic2admin),
                                                         get_scene_config(AIT_scenes.admin2aic)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000
             ),
        dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=1800,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.ait_football_inner),
                                                         get_scene_config(AIT_scenes.ait_football_outer),
                                                         get_scene_config(AIT_scenes.aic2admin),
                                                         get_scene_config(AIT_scenes.admin2aic)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000
             ),]