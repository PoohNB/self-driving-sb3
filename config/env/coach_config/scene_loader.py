from config.env.coach_config.dummy import dummy_placer
from config.env.reward_config import reward_mask_base



def get_scene_config(config):
    obsc_config = dict(car_obsc = config.placer_car,
                        ped_obsc = dummy_placer)
    
    reward_config = config.reward_config
    reward_config['config']['value_setting'] = reward_mask_base

    return dict(spawn_points = config.spawn,
                          cmd_config = config.cmd_points,
                          **obsc_config,
                          rewarder_config = config.reward_config)

from config.env.coach_config.command import cmd_guide

def get_coach_base(map_scenes):

    return  dict( parking_area=map_scenes.rest_area.parking_available,
                        ped_area=map_scenes.rest_area.people_area,
                        cmd_guide=cmd_guide
                        )