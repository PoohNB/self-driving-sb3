
from environment.tools import rewarder
import numpy as np
import random
from environment.tools.scene_designer import VehiclePlacer,PedestriansPlacer
from environment.tools import high_level_cmd
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
        self.total_score = 1
        # self.prev_scene = -1
        self.curr_scene=0
        # self.prev_sp_idx = -1
        self.sp_idx=0
        self.step =1        
        # self.scene_weights = [[(1/(self.total_score/self.step))]*len(cf['spawn_points']) for cf in self.scene_configs]
        self.cmd_decode = cmd_guide['cmd_dict']
        self.eval_mode_activated = False
        
        # spawn transform from config
        self.spawn_trans = []
        self.num_spawns = []
        for cf in self.scene_configs:
            spawn_points = [get_correspond_waypoint_transform(carla_map,sp) for sp in cf['spawn_points']]
            self.num_spawns.append(len(spawn_points))
            self.spawn_trans.append(spawn_points)

        # get vehicle placer and pedestrian placer
        vehicle_configs = [cf['car_obsc'] for cf in self.scene_configs if cf['car_obsc']]
        self.vehicle_placer = VehiclePlacer(self.world,vehicle_configs,parking_area) 
        pedestrian_configs = [cf['ped_obsc'] for cf in self.scene_configs]          
        self.pedestrian_placer = PedestriansPlacer(self.world,pedestrian_configs,ped_area)
        # rewarder list

        rewarder_configs = [cf['rewarder_config'] for cf in self.scene_configs]
        
        self.rewarders = []
        for cf in rewarder_configs:
            rewarder_class = getattr(rewarder,cf['name'])
            self.rewarders.append(rewarder_class(**cf['config'],
                                      vehicle=self.car,
                                      collision_sensor = self.colli_sensor) )
                
        # cmd pillar
        default_cmd = cmd_guide['default_cmd']
        commander_configs = [cf['cmd_config'] for cf in self.scene_configs]
        self.command_points=[]
        for cf in commander_configs:
            command_points_class = getattr(high_level_cmd,cf['name'])
            self.command_points.append(command_points_class(cf['configs'],default_cmd))

    def get_len_maneuver(self):

        return len(list(self.cmd_decode.values())[0])
    
    def eval_mode(self):
        self.eval_mode_activated = True

    def random_mode(self):
        self.eval_mode_activated = False

    def reset(self):
        # select scene

        # const = 1 + (self.curr_scene == self.prev_scene)*0.2 + (self.prev_sp_idx == self.sp_idx)*0.1

        # score_weighted = 1/max(1,self.total_score)
        # self.scene_weights[self.curr_scene][self.sp_idx] = ((self.scene_weights[self.curr_scene][self.sp_idx])
        #                                                      +score_weighted)/2
        # print("scene weights :",self.scene_weights)
        # scene_weight = [sum(weights)/len(weights) for weights in self.scene_weights]
        # self.curr_scene = random.choices(range(self.num_scenes),weights=scene_weight,k=1)[0]
        if self.eval_mode_activated:

            self.sp_idx+=1

            if self.num_spawns[self.curr_scene]<= self.sp_idx:
                self.sp_idx = 0
                self.curr_scene=(self.curr_scene+1)%self.num_scenes
                
        else:
            self.curr_scene = random.choice(range(self.num_scenes))
            self.sp_idx = random.choice(range(len(self.spawn_trans[self.curr_scene])))
        # Set the scene
        self.vehicle_placer.reset(self.curr_scene)
        self.pedestrian_placer.reset(self.curr_scene)
        # reset rewarder
        self.info = self.rewarders[self.curr_scene].reset()
        # reset high level command
        self.maneuver=self.command_points[self.curr_scene].reset()
        #set the agent car on spawn points
        self.car.move(self.spawn_trans[self.curr_scene][self.sp_idx])
        self.info['maneuver'] = self.maneuver
        self.step =0

        # self.prev_sp_idx = self.sp_idx
        # self.prev_scene = self.curr_scene

        return self.cmd_decode[self.maneuver],self.info

    def review(self):

        """
        calculate maneuver , score , terminate,
        
        """
        self.maneuver = self.command_points[self.curr_scene](self.car.get_xy_location())

   
        self.score,self.terminate,self.info = self.rewarders[self.curr_scene](being_obstructed=False,maneuver=self.maneuver)
        self.step+=1
        self.total_score+=self.score
        self.info['maneuver'] = self.maneuver
        # translate to cmd
        return self.cmd_decode[self.maneuver],self.score,self.terminate,self.info

    def set_movement(self):
        pass




