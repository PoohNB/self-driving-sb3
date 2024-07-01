
from environment.tools.rewarder import rewarder_type
import numpy as np
import random
from environment.tools.scene_designer import VehiclePlacer,PedestriansPlacer
from environment.tools.high_level_cmd import high_level_cmd_type
import carla
from typing import List, Tuple, Dict
from environment.tools.utils import get_correspond_waypoint_transform
class Coach:

    """
    - manipulate spawn point, scene, obstrucle, reward object, maneuver

    """

    def __init__(self,
                 scene_configs: List[Dict],
                 parking_area: List[Tuple[float, float]],
                 ped_area: Tuple[Tuple[float, float], Tuple[float, float]],
                 cmd_guide,
                 env):
        self.env = env()
        self.world = self.env.world
        carla_map=self.world.get_map()
        self.car = self.env.car
        self.colli_sensor = self.env.colli_sensor
        self.scene_configs = scene_configs
        self.num_scenes = len(self.scene_configs)
        self.scene_scores = [[1]*len(cf['spawn_points']) for cf in self.scene_configs]
        self.total_score = 1
        self.curr_scene=0
        self.sp_idx=0
        self.cmd_decode = cmd_guide['cmd_dict']
        
        # spawn transform from config
        self.spawn_trans = [[get_correspond_waypoint_transform(carla_map,sp) for sp in cf['spawn_points']] for cf in self.scene_configs]
        # get vehicle placer and pedestrian placer
        vehicle_configs = [cf['car_obsc'] for cf in self.scene_configs if cf['car_obsc']]
        self.vehicle_placer = VehiclePlacer(self.world,vehicle_configs,parking_area) 
        pedestrian_configs = [cf['ped_obsc'] for cf in self.scene_configs]          
        self.pedestrian_placer = PedestriansPlacer(self.world,pedestrian_configs,ped_area)
        # rewarder list
        rewarder_configs = [cf['rewarder_config'] for cf in self.scene_configs]
        self.rewarders = [
            rewarder_type[cf['name']](**cf['config'],
                                      vehicle=self.car,
                                      collision_sensor = self.colli_sensor) 
                                      for cf in rewarder_configs
        ]
        # cmd pillar
        default_cmd = cmd_guide['default_cmd']
        commander_configs = [cf['cmd_config'] for cf in self.scene_configs]
        self.command_points = [

            high_level_cmd_type[cf['name']](cf['configs'],default_cmd)
                    for cf in commander_configs

        ]
    def get_len_maneuver(self):

        return len(list(self.cmd_decode.values())[0])

    def reset(self):
        # select scene
        self.scene_scores[self.curr_scene][self.sp_idx] = max(1,self.total_score)
        scene_weight = [1/sum(scores) for scores in self.scene_scores]
        self.curr_scene = random.choices(range(self.num_scenes),weights=scene_weight,k=1)[0]
        # Set the scene
        self.vehicle_placer.reset(self.curr_scene)
        self.pedestrian_placer.reset(self.curr_scene)
        # reset rewarder
        self.info = self.rewarders[self.curr_scene].reset()
        # reset high level command
        self.maneuver=self.command_points[self.curr_scene].reset()
        #set the agent car on spawn points
        spawn_weight = [1/score for score in self.scene_scores[self.curr_scene]]
        self.sp_idx = random.choices(range(len(self.spawn_trans[self.curr_scene])),weights=spawn_weight,k=1)[0]
        self.car.move(self.spawn_trans[self.curr_scene][self.sp_idx])
        self.total_score = 1
        self.info['maneuver'] = self.maneuver

        return self.cmd_decode[self.maneuver],self.info

    def review(self):

        """
        calculate maneuver , score , terminate,
        
        """
        self.maneuver = self.command_points[self.curr_scene](self.car.get_xy_location())

   
        self.score,self.terminate,self.info = self.rewarders[self.curr_scene](being_obstructed=False,maneuver=self.maneuver)
  
        self.total_score+=self.score
        self.info['maneuver'] = self.maneuver
        # translate to cmd
        return self.cmd_decode[self.maneuver],self.score,self.terminate,self.info

    def set_movement(self):
        pass






    # def get_random_color(self):
    #     r = random.randint(0, 255)
    #     g = random.randint(0, 255)
    #     b = random.randint(0, 255)
    #     return f'{r},{g},{b}'
    #     # if self.random_color: 
    #     #     for veh in self.veh_list:
    #     #         veh.set_attribute('color', self.get_random_color())

    #     # for attr in blueprint:
    #     # if attr.is_modifiable:
    #     #     blueprint.set_attribute(attr.id, random.choice(attr.recommended_values))