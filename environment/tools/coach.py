
from environment.tools.rewarder import rewarder_type
import numpy as np
import random
from environment.tools.scene_designer import VehiclePlacer,PedestriansPlacer
from environment.tools.high_level_cmd import high_level_cmd_type
import carla
from typing import List, Tuple, Dict

class Coach:

    """
    - manipulate spawn point, scene, obstrucle, reward object, maneuver

    """

    def __init__(self,
                 scene_configs: List[Dict],
                 parking_area: List[Tuple[float, float]],
                 ped_area: Tuple[Tuple[float, float], Tuple[float, float]],
                 env):
        self.env = env()
        self.world = self.env.world
        self.car = self.env.car
        self.colli_sensor = self.env.colli_sensor
        self.scene_configs = scene_configs
        self.num_scenes = len(self.scene_configs)
        
        # spawn transform from config
        self.spawn_trans = [self.carla_list_transform(cf['spawn_points']) for cf in self.scene_configs]
        # get vehicle placer and pedestrian placer
        vehicle_configs = [cf['car_obsc'] for cf in self.scene_configs]
        pedestrian_configs = [cf['ped_obsc'] for cf in self.scene_configs]
        self.vehicle_placer = VehiclePlacer(self.world,vehicle_configs,parking_area)   
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
        commander_configs = [cf['cmd_config'] for cf in self.scene_configs]
        self.command_points = [

            high_level_cmd_type[cf['name']](cf['cmd_configs'])
                    for cf in commander_configs

        ]

    def carla_list_transform(self, area:List) -> carla.Transform:
        """Convert a configuration dictionary into a CARLA transform."""
        return [carla.Transform(carla.Location(*config['Location']), carla.Rotation(*config['Rotation'])) for config in area]
    

    def reset(self):
        # select scene
        self.curr_scene = random.choice(range(self.num_scenes))
        # Set the scene
        self.vehicle_placer.reset(self.curr_scene)
        self.pedestrian_placer.reset(self.curr_scene)
        # reset rewarder
        self.rewarders[self.curr_scene].reset()
        # reset high level command
        self.maneuver=self.command_points[self.curr_scene].reset()
        #set the agent car on spawn points
        sp = random.choice(self.spawn_trans[self.curr_scene])
        self.car.move(sp)

        return np.array([self.maneuver])

    def review(self):

        """
        calculate maneuver , score , terminate,
        """
        self.score,self.terminate,self.reason = self.rewarders[self.curr_scene]()

        self.maneuvar = self.command_points[self.curr_scene](self.car.get_xy_location())

        return np.array([self.maneuver]),self.score,self.terminate,self.reason

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