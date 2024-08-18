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

def norm_dist(x,s=0.01):
    return np.exp((-x**2)/s)

def sigmoid(x,c=0.5,s=10):
    return 1/(1+np.exp(-s*(x-c)))

def centrifugal_force(v,w,m=1):
    return m*v*w


class RewardMaskPathV0:

    """
    no reward only terminate when collis
    """

    def __init__(self,
                  mask_path,
                  vehicle,
                  collision_sensor,
                  value_setting,
                  end_point=None):
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
        self.end_point = end_point

        self.color = "blue"
        self.reward_scale = value_setting['reward_scale']
        self.out_of_road_count_limit = value_setting['out_of_road_count_limit']
        self.staystill_limit = value_setting['staystill_limit']
        self.max_velo = value_setting['max_velo']
        self.max_distance = self.max_velo*value_setting['step_time']
        self.max_steer = value_setting['max_steer']
        self.minimum_distance = value_setting['minimum_distance']
        self.mid_steer_range = value_setting['mid_steer_range']
        self.max_angular_velo =  value_setting['max_angular_velo']

    def reset(self):
        self.started=False
        self.staystill_count = 0
        self.out_of_road_count = 0        
        self.total_distance = 0        
        self.step = 0
        self.cF=0
        self.angular_velo=0
        self.terminate = False
        self.reason = ""

        self.prev_position = None
        self.curr_steer = 0
        self.velo = 0
        self.norm_velo = 0
        self.norm_distance = 0

        self.add_reset()

        # self.prev_area = 1 # 1 is blue 2 is black 3 is red 4 is green
        return self.get_info()
    
    def add_reset(self):
        self.prev_steer_side = "forward"

    def _get_car_position_on_map(self, car_position):
        # Convert car position from CARLA coordinates to image coordinates
        x, y = car_position
        img_x = int(self.ref_point[1] +x * self.m)
        img_y = int(self.ref_point[0] +y * self.m)
        return img_x, img_y

    def get_info(self):
        return {"reason":self.reason,
                "color":self.color,
                "norm velocity":f"{self.norm_velo:.3f}",
                "norm distance":f"{self.norm_distance:.3f}",
                "centrifugal force":f"{self.cF:.2f}"}
    
    def check_status(self):
        # update prev value
        if self.prev_position is None:
            self.car_pos = self.car.get_location()
        self.prev_position = (self.car_pos.x,self.car_pos.y)
        self.prev_steer = self.curr_steer
        self.prev_velo = self.velo

        # check distance
        self.car_pos = self.car.get_location()
        self.distance = math.sqrt((self.car_pos.x-self.prev_position[0])**2+(self.car_pos.y-self.prev_position[1])**2)
        if self.end_point is not None:
            if math.sqrt((self.car_pos.x-self.end_point[0])**2+(self.car_pos.y-self.end_point[1])**2)<3:
                self.terminate=True
        self.norm_distance = (self.distance/self.max_distance)
        self.total_distance+=self.distance

        # check steer
        self.curr_steer = self.car.get_control().steer
        self.steer_angle_change = self.curr_steer - self.prev_steer # [0,2]

        # check velocity
        self.velo = self.car.get_xy_velocity()
        self.norm_velo = (self.velo/self.max_velo) 
        self.velo_change = abs(self.velo-self.prev_velo)

        # self.angular_velocity
        self.angular_velo = self.car.get_angular_velocity().z
        self.norm_angular_velo = (self.angular_velo/self.max_angular_velo)
        self.cF = centrifugal_force(v=self.velo,w=self.angular_velo)

    def collision_check(self):
        if self.colli.collision:
            self.terminate = True
            self.reason = "collision terminate"

    def reward_fn(self):  
 
        self.collision_check()
            
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
        # terminate when coli with something that not road or road label
        self.args = args
        self.step +=1
        # initial reward
        self.reward = 0
        # get all information
        self.check_status()

        self.reward_fn()      

        return self.reward,self.terminate,self.get_info()
    




class RewardMaskPathV1(RewardMaskPathV0):
    """
    for straight or curve (not complex scenario)
    - give reward on move within blue mask
    - give only parse negative reward on termination from bad behavior
    
    """

    def __init__(self,
                  mask_path,
                  vehicle,
                  collision_sensor,
                  value_setting,
                  end_point=None):
        
        super().__init__(mask_path,
                        vehicle,
                        collision_sensor,
                        value_setting,
                        end_point=end_point)
        
    def reward_fn(self):

        self.being_obstructed = self.args['being_obstructed']
        self.maneuver = self.args['maneuver']

        self.collision_check()
        self.stay_still_check()
        self.out_path_check()
        self.steer_check()

    def steer_check(self):

        # determine steer
        if self.curr_steer < -(self.mid_steer_range/2):
            self.steer_side = "left"
        elif self.curr_steer>(self.mid_steer_range/2):
            self.steer_side = "right"
        else:
            self.steer_side = "forward"

        if self.color == "blue":
            # jerk shaky check
            if self.steer_side != self.prev_steer_side and self.prev_steer_side != "forward" and self.steer_side != "forward":
                self.reward -= self.reward_scale*0.45

            # small reward when can keep to steer at 0 for make it try to keep it strigth
        
            if self.maneuver=="right" and self.steer_side=="left":
                self.reward -=   self.reward_scale *0.3
            elif self.maneuver=="left"and self.steer_side=="right":
                self.reward -=  self.reward_scale *0.3
            elif self.maneuver=="forward":
    
                if (self.steer_side=="forward"):
                    self.reward += norm_dist(self.curr_steer,s=0.02) * self.reward_scale *0.45 # 0.6,0.3 change s to 0.01 for better focus
                    if self.prev_steer_side == "forward":
                        self.reward -= (abs(self.steer_angle_change)/(self.mid_steer_range))*self.reward_scale*0.3 # 0.3 0.15
                    else:
                        self.reward -= self.reward_scale*0.3 # 0.3 0.15
                else:
                    self.reward -= self.reward_scale*0.3 # 0.3 0.15

        else:
            self.reward -= norm_dist(self.curr_steer) * self.reward_scale *0.3 # 0.3 0.15

    def collision_check(self):
        # 1+1
        if self.colli.collision:
            self.terminate = True
            self.reason = "collision terminate"
            self.reward-= (1+self.norm_velo)*self.reward_scale

    def stay_still_check(self):
        # if car move too slow
        if self.distance < self.minimum_distance and self.being_obstructed:
            self.reward += self.reward_scale*1.5 # 1.5
        elif self.distance < self.minimum_distance:
            if self.step>15:
                self.started=True
            if self.started:
                self.staystill_count+=1
                self.reward -= self.reward_scale*1.5 # 1.5
        else:
            self.started = True
            self.staystill_count=0


        if self.staystill_count>self.staystill_limit:
            self.terminate = True
            self.reward -= self.reward_scale*5
            self.reason = "stay still for too long"

    def out_path_check(self):
        # determine position  mask
        # blue reward => norm_velo
        # red reward => -(norm_velo)
        # black reward => -norm_velo*0.6
        # green reward => -norm_velo*0.2
        img_x, img_y = self._get_car_position_on_map((self.car_pos.x, self.car_pos.y))

        if 0 <= img_x < self.route.shape[1] and 0 <= img_y < self.route.shape[0]:
            pixel_value = self.route[img_y, img_x]
            
            # Blue area reward
            if (pixel_value == [255, 0, 0]).all():
                self.color = "blue"

                self.reward += self.norm_distance*self.reward_scale
                self.out_of_road_count=0
     
             # Red area penalty
            elif (pixel_value == [0, 0, 255]).all():
                self.color = "red"
                self.reward -= self.norm_distance*self.reward_scale*0.8

                self.out_of_road_count+=2
            # black area
            elif (pixel_value == [0, 0, 0]).all():
                self.reward -= self.norm_distance*0.6*self.reward_scale
                self.color = "black"

                self.out_of_road_count+=1

            # green area
            elif (pixel_value == [0, 255, 0]).all():
                self.reward -= self.norm_distance*0.2*self.reward_scale
                self.color = "green"

        if self.out_of_road_count > self.out_of_road_count_limit:
            self.terminate = True
            self.reason = "out of the path for too long"
            self.reward -= self.reward_scale*5

class RewardMaskPath_Backup:

    """
    work for speed 15 km/hr
    """

    def __init__(self,
                  mask_path,              
                  vehicle,
                  collision_sensor,
                  value_setting,
                  end_point=None):
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
        self.total_distance = 0
        self.step = 0
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
        self.step +=1
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
        

        return self.reward,self.terminate,self.get_info()
    
    def get_info(self):
        return {"reason":self.reason,
                "color":self.color,
                "angular velocity":f"{self.car.get_angular_velocity().z:.3f}",
                "centrifugal force":f"{centrifugal_force(v=self.car.get_xy_velocity(),w=self.car.get_angular_velocity().z):.2f}"}
    
    def check_status(self):
        self.car_pos = self.car.get_location()
        if self.prev_position is None:
            self.prev_position = (self.car_pos.x,self.car_pos.y)
        self.distance = math.sqrt((self.car_pos.x-self.prev_position[0])**2+(self.car_pos.y-self.prev_position[1])**2)
        self.prev_position = (self.car_pos.x,self.car_pos.y)
        self.total_distance+=self.distance

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
            if self.step>15:
                self.started=True
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
    


    
 