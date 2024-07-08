from config.env.coach_config.AIT_scenes import ait_football_inner,ait_football_outer
from config.env.coach_config.AIT_scenes.rest_area import parking_available,people_area
from config.env.coach_config.dummy import dummy_placer
from config.env.coach_config.command import cmd_guide

no_obsc = dict(car_obsc = dummy_placer,
               ped_obsc = dummy_placer)

full_obsc_outer =  dict(car_obsc = ait_football_outer.placer_car,
                        ped_obsc = dummy_placer)

full_obsc_inner = dict(car_obsc = ait_football_inner.placer_car,
                        ped_obsc = dummy_placer)

scene_outer_easy = dict(spawn_points = ait_football_outer.spawn_short,
                          cmd_config = ait_football_outer.cmd_points,
                          **no_obsc,
                          rewarder_config = ait_football_outer.reward_config)

scene_outer_med = dict(spawn_points = ait_football_outer.spawn_long,
                          cmd_config = ait_football_outer.cmd_points,
                          **no_obsc,
                          rewarder_config = ait_football_outer.reward_config)

scene_outer_hard = dict(spawn_points = ait_football_outer.spawn_long,
                          cmd_config = ait_football_outer.cmd_points,
                           **full_obsc_outer,
                          rewarder_config = ait_football_outer.reward_config)

scene_inner_easy = dict(spawn_points = ait_football_inner.spawn_short,
                          cmd_config = ait_football_inner.cmd_points,
                          **no_obsc,
                          rewarder_config = ait_football_inner.reward_config)

scene_inner_med = dict(spawn_points = ait_football_inner.spawn_long,
                          cmd_config = ait_football_inner.cmd_points,
                          **no_obsc,
                          rewarder_config = ait_football_inner.reward_config)

scene_inner_hard = dict(spawn_points = ait_football_inner.spawn_long,
                          cmd_config = ait_football_inner.cmd_points,
                          **full_obsc_inner,
                          rewarder_config = ait_football_inner.reward_config)


coach_base = dict( parking_area=parking_available,
                    ped_area=people_area,
                    cmd_guide=cmd_guide
                    )