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

env_config_rand_cam = dict(**env_config_base,
                      rand_cams_pos = True)



levels = [dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=900,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.ait_football_inner),
                                                         get_scene_config(AIT_scenes.ait_football_outer),],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 100000,
             description = "ait football"
             ),
          dict(env= dict(**env_config_rand_cam,
                         carla_setting = carla_setting,
                         max_step=900,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.ait_football_inner),
                                                         get_scene_config(AIT_scenes.ait_football_outer),],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 100000,
             description = "ait football rand cams pos"
             ),
          dict(env= dict(**env_config_aug,
                         carla_setting = carla_setting,
                         max_step=900,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.ait_football_inner),
                                                         get_scene_config(AIT_scenes.ait_football_outer),],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 100000,
             description = "ait football rand cams pos"
             ),


        dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=550,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.admin2aic),
                                                         get_scene_config(AIT_scenes.aic2admin)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000,
             description = "aic-admin"
             ),

        dict(env= dict(**env_config_rand_cam,
                         carla_setting = carla_setting,
                         max_step=550,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.admin2aic),
                                                         get_scene_config(AIT_scenes.aic2admin)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000,
             description = "admin-aic"
             ),

        dict(env= dict(**env_config_aug,
                         carla_setting = carla_setting,
                         max_step=550,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.admin2aic),
                                                         get_scene_config(AIT_scenes.aic2admin)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000,
             description = "admin-aic"
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
             total_timesteps = 200000,
             description = "all"
             ),
             dict(env= dict(**env_config_aug,
                         carla_setting = carla_setting,
                         max_step=1800,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.ait_football_inner),
                                                         get_scene_config(AIT_scenes.ait_football_outer),
                                                         get_scene_config(AIT_scenes.aic2admin),
                                                         get_scene_config(AIT_scenes.admin2aic)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000,
             description = "all,shaky camera"
             ),
        dict(env= dict(**env_config_base,
                         carla_setting = carla_setting,
                         max_step=2500,
                        coach_config=dict(scene_configs=[get_scene_config(AIT_scenes.aic2admin),
                                                         get_scene_config(AIT_scenes.admin2aic)],
                                        **get_coach_base(AIT_scenes),)
                        ),
             total_timesteps = 200000,
             description = "admin and aic"
             ),]