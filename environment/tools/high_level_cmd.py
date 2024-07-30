

from typing import List, Tuple, Dict
import random
import numpy as np
from environment.tools.utils import calculate_distance

class DirectionCmd:
    """
    give the direction base on setting command of each point
    """


    def __init__(self, cmd_configs: List,default_cmd):
        self.cmd_configs = cmd_configs
        self.default_direct = default_cmd
    def reset(self):
        self.check_in = False
        self.curr_cmd = self.default_direct
        self.activate_pillar = None
        self.spawn_inside = True
        return self.curr_cmd

    def __call__(self, curr_pos: Tuple[float, float]):
            
        if self.activate_pillar is None:
            for i,cf in enumerate(self.cmd_configs):
                self.pillar_loc = cf['loc']
                distance = calculate_distance(self.pillar_loc, curr_pos)
                self.inner_rad, self.outer_rad = cf['call_rad']
                if distance < self.outer_rad:
                    self.rand_number = random.uniform(0, self.outer_rad - self.inner_rad)
                    self.activate_pillar = i
                    self.pillar_cmd = cf['cmd']
                    self.previous_distance = distance
                    return self.curr_cmd
            self.spawn_inside = False
        else:            
            # self.cmd_configs[self.activate_pillar]

            distance = calculate_distance(self.pillar_loc, curr_pos)
            
            if self.spawn_inside:
                if (self.previous_distance - distance) < -0.01:
                    self.check_in=True
                    self.spawn_inside = False
                elif (self.previous_distance - distance) > 0.01:
                    self.spawn_inside = False

            elif not self.check_in:

                if distance < self.inner_rad + self.rand_number:
                    self.curr_cmd = self.pillar_cmd

                if distance < self.inner_rad:
                    self.rand_number = random.uniform(0, self.outer_rad - self.inner_rad)
                    self.check_in = True

            else:
            
                if distance > self.inner_rad + self.rand_number:
                    self.curr_cmd = self.default_direct
                
                if distance > self.outer_rad:
                    self.activate_pillar = None
                    self.check_in = False



        return self.curr_cmd
    
