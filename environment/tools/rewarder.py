import cv2
import numpy as np
import os
import json
import math


def laplace_dist(x,b=0.2):
    return np.exp(-abs(x)/b)

def norm_dist(x,b=0.1):
    return np.exp(-x**2/b)

def diffsigmoid_decay(x):
    return 1/(1+np.exp(40*((x**2)-0.1)))

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

    def __init__(self, mask_path):
        # reference point is position 0,0 in carla
        # scale is meter/pixel
        scale_path = os.path.join(os.path.dirname(mask_path),"scale.json")
        with open(scale_path, 'r') as file:
            data = json.load(file)
        self.m = data['scale']
        self.ref_point = data['ref_point']
        self.route = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        self.prev_steer = 0
        self.terminate = False
        self.reason = ""
        self.colli = None
        self.out_of_road_count_limit = 30
        self.staystill_limit = 25
        self.reward_scale = 4
        self.minimum_distance = 0.015

    def apply_car(self,carla_car):
        self.car = carla_car

    def reset(self):
        self.started=False
        self.staystill_count = 0
        self.out_of_road_count = 0
        self.terminate = False
        self.reason = ""
        self.prev_position = None
        # self.prev_area = 1 # 1 is blue 2 is black 3 is red 4 is green

    def _get_car_position_on_map(self, car_position):
        # Convert car position from CARLA coordinates to image coordinates
        x, y = car_position
        img_x = int(self.ref_point[1] +x * self.m)
        img_y = int(self.ref_point[0] +y * self.m)
        return img_x, img_y
    
    def apply_collision_sensor(self,sensor):
        self.colli = sensor

    def __call__(self, being_obstructed=False):
        """
        condition 
        - stay still
        - forward
        - turning
        """
        if self.colli is None:
            raise Exception("not apply collision sensor yet")

        # terminate when coli with something that not road or road label
        if self.colli.collision:
            self.terminate = True
            self.reason = "collision terminate"
            return -20

        # initial reward
        reward = 0
        # get the car information for shape the reward
        # yaw = self.car.get_transform().rotation.yaw
        # max speed should be around 4.17 --> ~ 0.8 per step maximum
        car_position = self.car.get_location()
        if self.prev_position is None:
            self.prev_position = (car_position.x,car_position.y)
        distance = math.sqrt((car_position.x-self.prev_position[0])**2+(car_position.y-self.prev_position[1])**2)
        self.prev_position = (car_position.x,car_position.y)

        # if car not move to slow
        if distance < self.minimum_distance and being_obstructed:
            reward += 3
        elif distance < self.minimum_distance:
            self.staystill_count+=1
            if self.started:
                reward -= 3
        else:
            self.started = True
            self.staystill_count=0

        if self.staystill_count>self.staystill_limit and self.started:
            self.terminate = True
            self.reason = "stay still for too long"
            return -10
  
        # determine position - stay in blue reward out of blue panish
        img_x, img_y = self._get_car_position_on_map((car_position.x, car_position.y))

        if 0 <= img_x < self.route.shape[1] and 0 <= img_y < self.route.shape[0]:
            pixel_value = self.route[img_y, img_x]
            
            # Blue area reward
            if (pixel_value == [255, 0, 0]).all():
                reward += distance * self.reward_scale
                self.out_of_road_count=0
                # reward -= car_angle_change  * 2
                # reward for still angle depend on distance, car_angle_change maximum is 2 so don't worry
                # reward += diffsigmoid_decay(car_angle_change) * distance * self.reward_scale * 0.6 # <-- [0,1] maximum rate 0.5 2 is maximum angle 
                # reward += diffsigmoid_decay(abs(self.car.get_control().steer)) * distance * self.reward_scale * 0.2 # <-- [0,1] maximum rate 0.5 2 is maximum angle 

             # Red area penalty
            elif (pixel_value == [0, 0, 255]).all():
                reward -= distance * self.reward_scale
                self.out_of_road_count+=3
            # black area
            elif (pixel_value == [0, 0, 0]).all():
                reward -= distance * self.reward_scale * 0.5
                self.out_of_road_count+=1
   

        if self.out_of_road_count > self.out_of_road_count_limit:
            self.terminate = True
            self.reason = "out of the path for too long"

        # determine steer
        curr_steer = self.car.get_control().steer
        car_angle_change = abs(curr_steer - self.prev_steer)
        self.prev_steer = curr_steer
        
        # panish should lower than get car out of path, expect most value is 0.4 so it already lower than out of road punish 
        reward -= car_angle_change * distance * self.reward_scale

        # small reward when can keep to steer at 0 for make it try to keep it strigth
        reward += norm_dist(curr_steer) * distance * self.reward_scale * 0.4

        return reward
    
    def get_terminate(self):

        return self.terminate,self.reason
    
class RewardCoins:

    def __init__(self,):
        pass



rewarder_type = {'RewardDummy':RewardDummy,
                 'RewardPath':RewardPath,
                 'RewardCoins':RewardCoins} 

    # def reward_pos(self):

    #     curr_pos = self.car.get_transform()
    #     x = round((curr_pos.location.x * self.m) + self.ref_point[1])
    #     y = round((curr_pos.location.y * self.m) + self.ref_point[0])

    #     yaw = -curr_pos.rotation.yaw
    #     img_car = np.ones((100, 100, 3), dtype=np.uint8)*255
    #     start_point = (round(1.43*self.m)+50, round(0.61*self.m)+50)
    #     end_point = (round(-1.73*self.m)+50, round(-0.61*self.m)+50)
    #     img_car = cv2.rectangle(img_car, start_point, end_point, (0, 0, 0), -1)
    #     img_car = ndimage.rotate(img_car, yaw, order=0, reshape=0, mode='nearest')

    #     img_car = cv2.bitwise_not(img_car[:, :, 0])/255
    #     img_road = self.reward_mask[self.use_this_scene[self.sp]][y-50:y+50, x-50:x+50, 0]/255
    #     try:
    #         bit_and = cv2.bitwise_and(img_car, img_road)
    #         dont_record = False
    #     except:
    #         dont_record = True
        
    #     if not dont_record:
    #         img_out_road = self.reward_mask[self.use_this_scene[self.sp]][y-50:y+50, x-50:x+50, 2]/255
    #         bit_out_road = cv2.bitwise_and(img_car, img_out_road)
    #         sum_all = np.sum(img_car)
    #         sum_and = np.sum(bit_and)
    #         sum_out = np.sum(bit_out_road)

    #         if self.sp in self.obs_location.keys():
    #             curr_pos_obs = self.obs_car[self.which_obs].get_transform()

    #             obs_range_mask = self.obs_class.curr_position(self.obstacles[self.which_obs][0], (curr_pos_obs.location.x, curr_pos_obs.location.y), curr_pos_obs.rotation.yaw)
    #             obs_range_mask = obs_range_mask[y-50:y+50, x-50:x+50]/255
    #             obs_bit_and = cv2.bitwise_and(img_car, obs_range_mask)
    #             sum_mask = np.sum(obs_bit_and)

    #         velo = self.car.get_velocity()
    #         velo_r, velo_phi = self.cart2pol(velo.x, velo.y)
    #         convert_phi = yaw+velo_phi
    #         real_x, real_y = self.pol2cart(velo_r, convert_phi)
    #         real_x = real_x*3.6/8
    #         reward_real_x = (real_x-0.2)*3

    #         if self.sp in self.obs_location.keys():
    #             if sum_mask >= 1:
    #                 reward -= 10
    #                 self.count_in_obs += 1
    #             else:
    #                 reward += reward_real_x
    #         else:
    #             reward += reward_real_x

    #         dis_end = []
    #         for n in range(len(self.end_points[self.sp])):
    #             dis_end.append(curr_pos.location.distance(self.end_points[self.sp][n]))

    #         if curr_pos.location.distance(self.end_points[self.sp][2]) > self.prev_dist:
    #             self.check_rev_dir.append(1)
    #         else:
    #             self.check_rev_dir.append(0)

    #         self.prev_dist = curr_pos.location.distance(self.end_points[self.sp][2])

    #         if sum(self.check_rev_dir) >= N_CHECK_REVERSE:
    #             end_this = True
    #         else:
    #             end_this = False

    #         if sum_out > 0 or self.collision or end_this:
    #             reward = -30
    #             done = True
    #             end = False

    #         elif min(dis_end) <= 2:
    #             reward += ((sum_and/sum_all)-1)*1
    #             if sum_and/sum_all < 0.98:
    #                 reward -= 1
    #             done = True
    #             end = True
                
    #         else:
    #             reward += ((sum_and/sum_all)-1)*1
    #             if sum_and/sum_all < 0.98:
    #                 reward -= 1
    #             done = False
    #             end = False







