

from typing import List, Tuple, Dict
import random

class DirectionCmd:
    """
    give the direction base on setting command of each point
    """


    def __init__(self, cmd_configs: List[Dict]):
        self.cmd_configs = cmd_configs
        for cf in self.cmd_configs:
            inner,outer = cf['call_rad']
            cf['call_rad2'] = (inner**2,outer**2)

    def reset(self):
        self.check_in = False
        self.curr_cmd = 0
        self.activate_pillar = None
        
        return self.curr_cmd

    def __call__(self, curr_pos: Tuple[float, float]):
            
        if self.activate_pillar is None:
            for i,cf in enumerate(self.cmd_configs):
                self.pillar_loc = cf['loc']
                distance2 = self.calculate_distance_power2(self.pillar_loc, curr_pos)
                self.inner_rad2, self.outer_rad2 = cf['call_rad2']
                if distance2 < self.outer_rad2:
                    self.rand_number = random.uniform(0, self.outer_rad2 - self.inner_rad2)
                    self.activate_pillar = i
                    self.pillar_cmd = cf['cmd']
        else:            
            # self.cmd_configs[self.activate_pillar]

            distance2 = self.calculate_distance_power2(self.pillar_loc, curr_pos)

            if not self.check_in:

                if distance2 < self.inner_rad2 + self.rand_number:
                    self.curr_cmd = self.pillar_cmd

                if distance2 < self.inner_rad2:
                    self.rand_number = random.uniform(0, self.outer_rad2 - self.inner_rad2)
                    self.check_in = True

            else:
            
                if distance2 > self.inner_rad2 + self.rand_number:
                    self.curr_cmd = 0
                
                if distance2 > self.outer_rad2:
                    self.activate_pillar = None
                    self.check_in = False
    
        return self.curr_cmd
    
    @staticmethod
    def calculate_distance_power2(prev_pos:tuple,cur_pos:tuple):
        """Calculate Euclidean distance between two positions."""
        return (prev_pos[0] - cur_pos[0])**2 + (prev_pos[1] - cur_pos[1])**2

high_level_cmd_type={"DirectionCmd":DirectionCmd}