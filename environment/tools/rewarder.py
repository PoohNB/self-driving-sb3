import cv2
import numpy as np
import os
import json
import math

"""
when we drive car, 
in straight way, 
mostly we will keep the steer at 0,
when the car is near the edge of the road we move steer a little to keep the lane

in intersection,
we use the bigger steer angle to change the lane
"""


def laplace_dist(x,b=0.2):
    return np.exp(-abs(x)/b)

def norm_dist(x,b=0.01):
    return np.exp((-x**2)/b)

def diffsigmoid_decay(x):
    return 1/(1+np.exp(40*((x**2)-0.1)))

def shift_sigmoid(x):
    return 1/(1+np.exp(-12*(x-0.6)))

class RewardDummy:

    def __init__(self):

        pass

    def apply_car(self,carla_car):
        pass

    def apply_collision_sensor(self,sensor):
        pass

    def reward(self,car,being_obstructed):

        return 0
    
    def get_terminate(self):

        return False
    
    def reset(self):
        pass
    
class RewardPath:

    def __init__(self,
                  mask_path,
                  vehicle,
                  collision_sensor):
        # reference point is position 0,0 in carla
        # scale is meter/pixel
        scale_path = os.path.join(os.path.dirname(mask_path),"scale.json")
        with open(scale_path, 'r') as file:
            data = json.load(file)
        self.m = data['scale']
        self.ref_point = data['ref_point']
        self.route = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        self.car = vehicle
        self.colli = collision_sensor

        self.color = "blue"
        self.out_of_road_count_limit = 20
        self.staystill_limit = 25
        self.reward_scale = 2
        self.steer_reward_scale = 0.5
        self.max_distance = 1
        # self.terminate_reward = -self.max_speed*((self.out_of_road_count_limit)/5)*self.reward_scale
        self.minimum_distance = 0.015
        self.mid_range = (-0.05,0.05)

    def reset(self):
        self.started=False
        self.staystill_count = 0
        self.out_of_road_count = 0
        self.terminate = False
        self.reason = ""
        self.prev_position = None
        self.prev_steer = 0
        self.prev_steer_side = 0
        # self.prev_area = 1 # 1 is blue 2 is black 3 is red 4 is green
        return self.get_info()

    def _get_car_position_on_map(self, car_position):
        # Convert car position from CARLA coordinates to image coordinates
        x, y = car_position
        img_x = int(self.ref_point[1] +x * self.m)
        img_y = int(self.ref_point[0] +y * self.m)
        return img_x, img_y

    def __call__(self, **args):
        """
        condition
        - collision - get terminate , -20 score 
        for the rest it will depend on distance
        - stay still - if the distance change to small it will count as stay still , -1 score every step
        - steering -

        - forward
        - turning
        """
        if self.colli is None:
            raise Exception("rewarder not apply collision sensor yet")

        being_obstructed = args['being_obstructed']
        maneuver = args['maneuver']
        # terminate when coli with something that not road or road label

        # initial reward
        self.reward = 0
        # get distance
        self.check_status()
        # check collision
        self.collision_check()
        # if car move too slow
        self.stay_still_check(being_obstructed)
        # make steer not too shakky
        self.steer_check(maneuver)
        # determine position - stay in blue reward out of blue panish
        self.out_path_check()
        
        # if self.terminate:
            # self.reward = self.terminate_reward

        return self.reward,self.terminate,self.get_info()
    
    def get_info(self):
        return {"reason":self.reason,
                "color":self.color}
    
    def check_status(self):
        self.car_pos = self.car.get_location()
        if self.prev_position is None:
            self.prev_position = (self.car_pos.x,self.car_pos.y)
        self.distance = math.sqrt((self.car_pos.x-self.prev_position[0])**2+(self.car_pos.y-self.prev_position[1])**2)
        self.prev_position = (self.car_pos.x,self.car_pos.y)

        # determine position - stay in blue reward out of blue panish
        img_x, img_y = self._get_car_position_on_map((self.car_pos.x, self.car_pos.y))

        if 0 <= img_x < self.route.shape[1] and 0 <= img_y < self.route.shape[0]:
            pixel_value = self.route[img_y, img_x]
            
            # Blue area reward
            if (pixel_value == [255, 0, 0]).all():
                self.color = "blue"
     
             # Red area penalty
            elif (pixel_value == [0, 0, 255]).all():
                self.color = "red"
            # black area
            elif (pixel_value == [0, 0, 0]).all():
                self.color = "black"

            elif (pixel_value == [0, 255, 0]).all():
                self.color = "green"

    def collision_check(self):
        if self.colli.collision:
            self.terminate = True
            self.reason = "collision terminate"
            self.reward -= ((self.car.get_xy_velocity()/5)+self.max_distance) * self.reward_scale
        
    def stay_still_check(self,being_obstructed):
        # if car move too slow
        if self.distance < self.minimum_distance and being_obstructed:
            self.reward += self.max_distance*self.reward_scale
        elif self.distance < self.minimum_distance:
            if self.started:
                self.staystill_count+=1
                self.reward -= self.max_distance*self.reward_scale*2
        else:
            self.started = True
            self.staystill_count=0

        if self.staystill_count>self.staystill_limit:
            self.terminate = True
            self.reward -= 10
            self.reason = "stay still for too long"
    
    def steer_check(self,maneuver):

        # determine steer
        self.curr_steer = self.car.get_control().steer
        if self.curr_steer < self.mid_range[0]:
            self.steer_side = "left"
        elif self.curr_steer>self.mid_range[1]:
            self.steer_side = "right"
        else:
            self.steer_side = "forward"

        # shaky check
        if self.steer_side != self.prev_steer_side and self.prev_steer_side != "forward" and self.steer_side != "forward":
            self.reward -= self.steer_reward_scale*1.5

        car_angle_change = abs(self.curr_steer - self.prev_steer)

        # small reward when can keep to steer at 0 for make it try to keep it strigth
        if self.color == "blue":
            if maneuver=="right" and self.steer_side=="left":
                self.reward -=   self.steer_reward_scale
            elif maneuver=="left"and self.steer_side=="right":
                self.reward -=  self.steer_reward_scale
            elif maneuver=="forward":
                if (self.steer_side=="forward"):
                    self.reward += norm_dist(self.curr_steer) * self.steer_reward_scale *2
                    self.reward -= (abs(car_angle_change)/(abs(self.mid_range[1]-self.mid_range[0])))*self.steer_reward_scale
                else:
                    self.reward -=  self.steer_reward_scale

        else:
            self.reward -= norm_dist(self.curr_steer) * self.steer_reward_scale

        self.prev_steer_side = self.steer_side
        self.prev_steer = self.curr_steer
        
    def out_path_check(self):
        # determine position - stay in blue reward out of blue panish

        # Blue area reward
        if self.color == "blue":
            self.reward += self.distance * self.reward_scale
            self.out_of_road_count=0
            
            # Red area penalty
        elif self.color == "red":
            self.reward -= self.distance * self.reward_scale * 0.8
            self.out_of_road_count+=2
            
        # black area
        elif self.color == "black":
            self.reward -= self.distance * self.reward_scale * 0.6
            self.out_of_road_count+=1
            
        elif self.color == "green":
            self.reward -= self.distance * self.reward_scale * 0.2
            

        if self.out_of_road_count > self.out_of_road_count_limit:
            self.terminate = True
            self.reason = "out of the path for too long"
            self.reward -= 5 * self.reward_scale





    
class RewardCoins:

    def __init__(self,):
        pass



rewarder_type = {'RewardDummy':RewardDummy,
                 'RewardPath':RewardPath,
                 'RewardCoins':RewardCoins} 


    # def __call__(self, **args):
    #     """
    #     condition
    #     - collision - get terminate , -20 score 
    #     for the rest it will depend on distance
    #     - stay still - if the distance change to small it will count as stay still , -1 score every step
    #     - steering -

    #     - forward
    #     - turning
    #     """
    #     if self.colli is None:
    #         raise Exception("rewarder not apply collision sensor yet")

    #     being_obstructed = args['being_obstructed']
    #     maneuver = args['maneuver']
    #     # terminate when coli with something that not road or road label
    #     if self.colli.collision:
    #         self.terminate = True
    #         self.reason = "collision terminate"
    #         return -20,self.terminate,self.get_info()

    #     # initial reward
    #     self.reward = 0
    #     # get distance
    #     self.get_distance()
    #     # if car move too slow
    #     self.stay_still_check(being_obstructed)
    #     # make steer not too shakky
    #     self.steer_check()
    #     # determine position - stay in blue reward out of blue panish
    #     self.out_path_check()


    #     return self.reward,self.terminate,self.get_info()
    
    # def get_info(self):
    #     return {"reason":self.reason,
    #             "color":self.color}
    
    # def get_mask_color(self):
    #     return self.color
    
    # def get_distance(self):
    #     self.car_pos = self.car.get_location()
    #     if self.prev_position is None:
    #         self.prev_position = (self.car_pos.x,self.car_pos.y)
    #     self.distance = math.sqrt((self.car_pos.x-self.prev_position[0])**2+(self.car_pos.y-self.prev_position[1])**2)
    #     self.prev_position = (self.car_pos.x,self.car_pos.y)
        
    
    # def stay_still_check(self,being_obstructed):
    #     # if car move too slow
    #     if self.distance < self.minimum_distance and being_obstructed:
    #         self.reward += 4
    #     elif self.distance < self.minimum_distance:
    #         self.staystill_count+=1
    #         if self.started:
    #             self.reward -= 4
    #     else:
    #         self.started = True
    #         self.staystill_count=0

    #     if self.staystill_count>self.staystill_limit and self.started:
    #         self.terminate = True
    #         self.reason = "stay still for too long"
    #         self.reward = -10
    
    # def steer_check(self):
    #     # determine steer
    #     self.curr_steer = self.car.get_control().steer
    #     car_angle_change = abs(self.curr_steer - self.prev_steer)
    #     self.prev_steer = self.curr_steer
        
    #     # panish should lower than get car out of path, the most range is 2 but
    #     # expect most value is 0.4 so it already lower than out of road punish 
    #     self.reward -= min(car_angle_change,0.4) * self.distance * self.reward_scale

    #     # small reward when can keep to steer at 0 for make it try to keep it strigth
    #     self.reward += norm_dist(self.curr_steer) * self.distance * self.reward_scale * 0.2

    # def out_path_check(self):
    #     # determine position - stay in blue reward out of blue panish
    #     img_x, img_y = self._get_car_position_on_map((self.car_pos.x, self.car_pos.y))

    #     if 0 <= img_x < self.route.shape[1] and 0 <= img_y < self.route.shape[0]:
    #         pixel_value = self.route[img_y, img_x]
            
    #         # Blue area reward
    #         if (pixel_value == [255, 0, 0]).all():
    #             self.reward += self.distance * self.reward_scale
    #             self.out_of_road_count=0
    #             self.color = "blue"
     
    #          # Red area penalty
    #         elif (pixel_value == [0, 0, 255]).all():
    #             self.reward -= self.distance * self.reward_scale * 0.8
    #             self.out_of_road_count+=2
    #             self.color = "red"
    #         # black area
    #         elif (pixel_value == [0, 0, 0]).all():
    #             self.reward -= self.distance * self.reward_scale * 0.6
    #             self.out_of_road_count+=1
    #             self.color = "black"

    #         elif (pixel_value == [0, 255, 0]).all():
    #             self.reward -= self.distance * self.reward_scale * 0.2
    #             self.color = "green"

   

    #     if self.out_of_road_count > self.out_of_road_count_limit:
    #         self.terminate = True
    #         self.reward -= 5
    #         self.reason = "out of the path for too long"

