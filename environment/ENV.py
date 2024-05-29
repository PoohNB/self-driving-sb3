from environment.tools.scene_designer import locate_obstacle, create_point

import carla
import cv2
from collections import deque
import random
import numpy as np
import copy
from scipy import ndimage

CARLA_SERVER_IP = 'localhost'

SEG_SIZE = 272
RGB_SIZE = 480

N_CHECK_REVERSE = 32 # If driving reverse in this number, will terminate

MAX_PLAY_STEP = 400

TRAIN_SCENE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
TEST_SCENE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

IN_CHANNLES = 8
N_LOOK_BACK = 8

MAX_THROTTLE = 0.406
STEERING_SPEED = 0.2

class CarlaEnv():
    def __init__(self):

        # connect to carla
        self.client = carla.Client(CARLA_SERVER_IP, 2000)
        self.client.set_timeout(120)

        self.world = self.client.get_world()
        
        blueprints = self.world.get_blueprint_library()

        self.bp_car = blueprints.filter('evt_echo_4s')[0]

        # [model, width, height]
        # 100 = 1 meter
        self.obstacles = [['a2', 167, 360],
                          ['tt', 198, 402],
                          ['grandtourer', 213, 421],
                          ['c3', 168, 390],
                          ['coupe', 206, 478]]
        
        self.obs_class = locate_obstacle()
        self.bp_obstacles = {}
        for n in range(len(self.obstacles)):
            self.bp_obstacles[self.obstacles[n][0]] = blueprints.filter(self.obstacles[n][0])[0]

            self.obs_class.add(self.obstacles[n][0], [self.obstacles[n][1], self.obstacles[n][2]], [400, 400, 50, 50])

        # Spawn point of obstacles
        # 8-12 are the scene number that will have obstacles
        # 8, 11, 12 are the same scenes, but different obstacle action
        self.obs_location = {}
        self.obs_location[8] = create_point([-81.4313, -31.5256, 0.2], [0, 355, 0])
        self.obs_location[9] = create_point([-263.3722, 6.6308, 0.2], [0, 172, 0])
        self.obs_location[10] = create_point([-357.5998, -98.0666, 0.2], [0, 275, 0])
        self.obs_location[11] = create_point([-81.4313, -31.5256, 0.2], [0, 355, 0])
        self.obs_location[12] = create_point([-81.4313, -31.5256, 0.2], [0, 355, 0])

        # 0 for golf cart
        # 1-5 for obstacles
        rest_points_ori = [[[-890, 190, 0.2], [0, 0, 0]],
                           [[-890, 180, 0.2], [0, 0, 0]],
                           [[-890, 170, 0.2], [0, 0, 0]],
                           [[-890, 160, 0.2], [0, 0, 0]],
                           [[-890, 150, 0.2], [0, 0, 0]],
                           [[-890, 140, 0.2], [0, 0, 0]]]
        
        self.rest_points = []
        for n in range(len(rest_points_ori)):
            self.rest_points.append(create_point(rest_points_ori[n][0], rest_points_ori[n][1]))

        # Spawn cars at the resting points
        self.car = self.world.spawn_actor(self.bp_car, self.rest_points[0])

        self.obs_car = []
        for n in range(len(self.obstacles)):
            self.obs_car.append(self.world.spawn_actor(self.bp_obstacles[self.obstacles[n][0]], self.rest_points[n+1]))

        self.seg_width = round(SEG_SIZE*16/9)
        self.bp_seg = blueprints.find('sensor.camera.semantic_segmentation')
        self.bp_seg.set_attribute('image_size_x', str(self.seg_width))
        self.bp_seg.set_attribute('image_size_y', str(SEG_SIZE))
        self.bp_seg.set_attribute('fov', '69')
        self.bp_seg.set_attribute('sensor_tick', '0.001')

        self.rgb_width = round(RGB_SIZE*16/9)
        self.bp_rgb = blueprints.find('sensor.camera.rgb')
        self.bp_rgb.set_attribute('image_size_x', str(self.rgb_width))
        self.bp_rgb.set_attribute('image_size_y', str(RGB_SIZE))
        self.bp_rgb.set_attribute('fov', '69')
        self.bp_rgb.set_attribute('sensor_tick', '0.001')

        self.bp_colli = blueprints.find('sensor.other.collision')

        self.seg_tmp_f = None
        self.seg_tmp_l = None
        self.seg_tmp_r = None
        self.seg_tmp_b = None
        self.rgb_tmp_f = None

        ## Spawn sensors 
        
        # camera front,right,left,back
        self.seg_cam_f = self.world.spawn_actor(self.bp_seg, carla.Transform(carla.Location(0.98, 0, 1.675), carla.Rotation(-12.5, 0, 0)), attach_to=self.car)
        self.seg_cam_f.listen(lambda data: self.process_seg_f(data))

        self.seg_cam_l = self.world.spawn_actor(self.bp_seg, carla.Transform(carla.Location(0, -0.61, 1.675), carla.Rotation(-30, -90, 0)), attach_to=self.car)
        self.seg_cam_l.listen(lambda data: self.process_seg_l(data))

        self.seg_cam_r = self.world.spawn_actor(self.bp_seg, carla.Transform(carla.Location(0, 0.61, 1.675), carla.Rotation(-30, 90, 0)), attach_to=self.car)
        self.seg_cam_r.listen(lambda data: self.process_seg_r(data))

        self.seg_cam_b = self.world.spawn_actor(self.bp_seg, carla.Transform(carla.Location(-0.98, 0, 1.675), carla.Rotation(-12.5, 180, 0)), attach_to=self.car)
        self.seg_cam_b.listen(lambda data: self.process_seg_b(data))

        # rgb front camera
        self.rgb_cam_f = self.world.spawn_actor(self.bp_rgb, carla.Transform(carla.Location(0.98, 0, 1.675), carla.Rotation(-12.5, 0, 0)), attach_to=self.car)
        self.rgb_cam_f.listen(lambda data: self.process_rgb_f(data))

        # get map
        self.map_img = cv2.imread('../../utils/map.png')

        # reward sensor 
        self.colli_sensor = self.world.spawn_actor(self.bp_colli, carla.Transform(), attach_to=self.car)
        self.colli_sensor.listen(self.collision_callback)

        # reward mask
        self.reward_mask = {}
        self.reward_mask['route_1.png'] = cv2.imread(f'../../utils/reward_mask/admin_to_aic/route_1.png')
        self.reward_mask['route_2.png'] = cv2.imread(f'../../utils/reward_mask/admin_to_aic/route_2.png')

        self.use_this_scene = ['route_1.png', 'route_2.png', 'route_1.png', 'route_2.png', 'route_1.png', 'route_1.png',
                               'route_2.png', 'route_2.png', 'route_2.png', 'route_1.png', 'route_1.png', 'route_2.png', 'route_2.png']

        # For checking the golf cart can hit road and lane marking
        self.check_seman_tags = [1, 24]

        # Point of spawning the golf cart at each scene
        # Scene 0-7 is original scene
        # Scene 8-12 is ?
        spawn_points_ori = [[[7.44, -32.22, 0.01], [0, 90, 0]],
                            [[-106.90, -27.22, 0.01], [0, 349, 0]],
                            [[-229.05, 0.58, 0.01], [0, 169, 0]],
                            [[-341.08, -34.89, 0.01], [0, 48, 0]],
                            [[-356.78, -71.65, 0.01], [0, 261, 0]],
                            [[-350.27, -330.02, 0.01], [0, 270, 0]],
                            [[-300.71, -365.62, 0.01], [0, 180, 0]],
                            [[-166.81, -364.69, 0.01], [0, 180, 0]],
                            [[-106.90, -27.22, 0.01], [0, 349, 0]],
                            [[-229.05, 0.58, 0.01], [0, 169, 0]],
                            [[-356.78, -71.65, 0.01], [0, 261, 0]],
                            [[-106.90, -27.22, 0.01], [0, 349, 0]],
                            [[-106.90, -27.22, 0.01], [0, 349, 0]]]
        

        end_points_ori = [[[-106.55, -29.08, 0], [-105.97, -27.45, 0], [-105.74, -25.70, 0], [-105.39, -23.96, 0], [-105.04, -22.21, 0]],
                          [[2.09, -33.27, 0], [9.19, -33.27, 0], [3.83, -33.15, 0], [5.70, -33.15, 0], [7.44, -33.15, 0]],
                          [[-339.68, -35.82, 0], [-340.84, -34.55, 0], [-342.12, -33.38, 0], [-343.52, -32.45, 0], [-345.38, -31.06, 0]],
                          [[-231.49, -4.18, 0], [-231.14, -2.44, 0], [-230.91, -0.81, 0], [-230.56, 1.04, 0], [-230.21, 2.79, 0]],
                          [[-353.52, -173.56, 0], [-351.78, -173.56, 0], [-350.27, -173.44, 0], [-348.64, -173.44, 0], [-346.78, -173.33, 0]],
                          [[-301.41, -371.21, 0], [-301.52, -369.11, 0], [-301.52, -367.37, 0], [-301.52, -365.74, 0], [-301.64, -363.76, 0]],
                          [[-352.24, -327.47, 0], [-350.27, -327.35, 0], [-348.52, -327.35, 0], [-347.01, -327.23, 0], [-345.38, -327.00, 0]],
                          [[-244.99, -370.97, 0], [-244.99, -369.23, 0], [-244.99, -367.37, 0], [-244.99, -365.74, 0], [-244.99, -363.99, 0]],
                          [[2.09, -33.27, 0], [9.19, -33.27, 0], [3.83, -33.15, 0], [5.70, -33.15, 0], [7.44, -33.15, 0]],
                          [[-339.68, -35.82, 0], [-340.84, -34.55, 0], [-342.12, -33.38, 0], [-343.52, -32.45, 0], [-345.38, -31.06, 0]],
                          [[-353.52, -173.56, 0], [-351.78, -173.56, 0], [-350.27, -173.44, 0], [-348.64, -173.44, 0], [-346.78, -173.33, 0]],
                          [[2.09, -33.27, 0], [9.19, -33.27, 0], [3.83, -33.15, 0], [5.70, -33.15, 0], [7.44, -33.15, 0]],
                          [[2.09, -33.27, 0], [9.19, -33.27, 0], [3.83, -33.15, 0], [5.70, -33.15, 0], [7.44, -33.15, 0]]]
        
        self.spawn_points = []
        for n in range(len(spawn_points_ori)):
            self.spawn_points.append(create_point(spawn_points_ori[n][0], spawn_points_ori[n][1]))
            
        self.end_points = []
        for n in range(len(end_points_ori)):
            end_points_tmp = []
            for m in range(len(end_points_ori[n])):
                end_points_tmp.append(create_point(end_points_ori[n][m]))
            self.end_points.append(end_points_tmp)


        self.play = False
        self.play_images = np.zeros([MAX_PLAY_STEP*len(TEST_SCENE)*2, RGB_SIZE, self.rgb_width, 3], dtype=np.uint8)
        self.play_images_seg = np.zeros([MAX_PLAY_STEP*len(TEST_SCENE)*2, SEG_SIZE, SEG_SIZE], dtype=np.uint8)

        self.location_buffer = []

        self.m = 8.596200822454035
        self.ref_point = (4005, 6864)

        self.seg_state_buffer_f = deque(maxlen=IN_CHANNLES)
        self.seg_state_buffer_l = deque(maxlen=IN_CHANNLES)
        self.seg_state_buffer_r = deque(maxlen=IN_CHANNLES)
        self.seg_state_buffer_b = deque(maxlen=IN_CHANNLES)
        self.action_state_buffer = deque(maxlen=N_LOOK_BACK)

        self.frame_play = 0

    def process_seg_f(self, data):
        img = np.array(data.raw_data)
        img = img.reshape((SEG_SIZE, self.seg_width, 4))
        img = img[:, :, 2]
        seg_tmp = np.zeros([SEG_SIZE, self.seg_width], dtype=np.uint8)

        seg_tmp[img==1] = 1 # Road
        seg_tmp[img==24] = 2 # RoadLines
        seg_tmp[img==12] = 3 # Pedestrians
        seg_tmp[img==13] = 3 # Rider
        seg_tmp[img==14] = 3 # Car
        seg_tmp[img==15] = 3 # Truck
        seg_tmp[img==16] = 3 # Bus
        seg_tmp[img==18] = 3 # Motorcycle
        seg_tmp[img==19] = 3 # Bicycle

        self.seg_tmp_f = cv2.resize(seg_tmp, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)

    def process_seg_l(self, data):
        img = np.array(data.raw_data)
        img = img.reshape((SEG_SIZE, self.seg_width, 4))
        img = img[:, :, 2]
        seg_tmp = np.zeros([SEG_SIZE, self.seg_width], dtype=np.uint8)

        seg_tmp[img==1] = 1 # Road
        seg_tmp[img==24] = 2 # RoadLines
        seg_tmp[img==12] = 3 # Pedestrians
        seg_tmp[img==13] = 3 # Rider
        seg_tmp[img==14] = 3 # Car
        seg_tmp[img==15] = 3 # Truck
        seg_tmp[img==16] = 3 # Bus
        seg_tmp[img==18] = 3 # Motorcycle
        seg_tmp[img==19] = 3 # Bicycle

        self.seg_tmp_l = cv2.resize(seg_tmp, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)

    def process_seg_r(self, data):
        img = np.array(data.raw_data)
        img = img.reshape((SEG_SIZE, self.seg_width, 4))
        img = img[:, :, 2]
        seg_tmp = np.zeros([SEG_SIZE, self.seg_width], dtype=np.uint8)

        seg_tmp[img==1] = 1 # Road
        seg_tmp[img==24] = 2 # RoadLines
        seg_tmp[img==12] = 3 # Pedestrians
        seg_tmp[img==13] = 3 # Rider
        seg_tmp[img==14] = 3 # Car
        seg_tmp[img==15] = 3 # Truck
        seg_tmp[img==16] = 3 # Bus
        seg_tmp[img==18] = 3 # Motorcycle
        seg_tmp[img==19] = 3 # Bicycle

        self.seg_tmp_r = cv2.resize(seg_tmp, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)

    def process_seg_b(self, data):
        img = np.array(data.raw_data)
        img = img.reshape((SEG_SIZE, self.seg_width, 4))
        img = img[:, :, 2]
        seg_tmp = np.zeros([SEG_SIZE, self.seg_width], dtype=np.uint8)

        seg_tmp[img==1] = 1 # Road
        seg_tmp[img==24] = 2 # RoadLines
        seg_tmp[img==12] = 3 # Pedestrians
        seg_tmp[img==13] = 3 # Rider
        seg_tmp[img==14] = 3 # Car
        seg_tmp[img==15] = 3 # Truck
        seg_tmp[img==16] = 3 # Bus
        seg_tmp[img==18] = 3 # Motorcycle
        seg_tmp[img==19] = 3 # Bicycle

        self.seg_tmp_b = cv2.resize(seg_tmp, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)

    def collision_callback(self, event):
        if event.other_actor.semantic_tags[0] not in self.check_seman_tags:
            self.collision = True

    def process_rgb_f(self, data):
        img = np.array(data.raw_data)
        img = img.reshape((RGB_SIZE, self.rgb_width, 4))
        self.rgb_tmp_f = img[:, :, 0:3].astype(np.uint8)

    def set_world(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.2
        settings.max_substeps = 16
        settings.max_substep_delta_time = 0.0125
        self.world.apply_settings(settings)

    def reset_world(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = 0
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.max_culling_distance = 0
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)

    def reset(self, sp=None):
        self.timestep = 0 # Step in each step
        self.curr_steer_position = 0
        self.count_in_obs = 0 # Step inside obstacle range
        self.collision = False # Check if hit an obstacle

        # Check reverse direction
        self.check_rev_dir = deque(maxlen=N_CHECK_REVERSE)
        for n in range(N_CHECK_REVERSE):
            self.check_rev_dir.append(0)

        if sp is None:
            self.sp = random.choice(TRAIN_SCENE)
        else:
            self.sp = sp

        self.set_world()

        self.car.set_transform(self.spawn_points[self.sp])

        self.seg_tmp_f = None
        self.seg_tmp_l = None
        self.seg_tmp_r = None
        self.seg_tmp_b = None
        self.rgb_tmp_f = None

        self.world.tick()

        if self.sp in self.obs_location.keys():
            if sp is None:
                self.which_obs = np.random.randint(len(self.obstacles))
            else:
                if sp == 8:
                    self.which_obs = 0
                elif sp == 9:
                    self.which_obs = 1
                elif sp == 10:
                    self.which_obs = 2
                elif sp == 11:
                    self.which_obs = 3
                elif sp == 12:
                    self.which_obs = 4

            self.obs_car[self.which_obs].set_transform(self.obs_location[self.sp])

        # Set previous position for checking reverse
        curr_pos = self.car.get_transform()
        self.prev_dist = curr_pos.location.distance(self.end_points[self.sp][2])

        while True:
            if self.seg_tmp_f is not None:
                seg_state_f = self.seg_tmp_f
                self.seg_tmp_f = None
                break

        while True:
            if self.seg_tmp_l is not None:
                seg_state_l = self.seg_tmp_l
                self.seg_tmp_l = None
                break

        while True:
            if self.seg_tmp_r is not None:
                seg_state_r = self.seg_tmp_r
                self.seg_tmp_r = None
                break

        while True:
            if self.seg_tmp_b is not None:
                seg_state_b = self.seg_tmp_b
                self.seg_tmp_b = None
                break
        
        # If play then record images
        if self.play:
            while True:
                if self.rgb_tmp_f is not None:
                    rgb_state_f = self.rgb_tmp_f
                    self.rgb_tmp_f = None

                    self.play_images_seg[self.frame_play] = seg_state_f
                    self.play_images[self.frame_play] = rgb_state_f
                    self.frame_play += 1

                    curr_pos = self.car.get_transform()
                    x = round((curr_pos.location.x * self.m) + self.ref_point[1])
                    y = round((curr_pos.location.y * self.m) + self.ref_point[0])
                    
                    self.location_buffer.append([x, y, self.sp])
                    break

        for _ in range(IN_CHANNLES):
            self.seg_state_buffer_f.append(seg_state_f)
            self.seg_state_buffer_l.append(seg_state_l)
            self.seg_state_buffer_r.append(seg_state_r)
            self.seg_state_buffer_b.append(seg_state_b)

        for _ in range(N_LOOK_BACK):
            self.action_state_buffer.append([0])

    def get_state(self):
        state = {}
        
        state_seg_tmp_f = np.array(self.seg_state_buffer_f, dtype=np.uint8)
        state_seg_tmp_l = np.array(self.seg_state_buffer_l, dtype=np.uint8)
        state_seg_tmp_r = np.array(self.seg_state_buffer_r, dtype=np.uint8)
        state_seg_tmp_b = np.array(self.seg_state_buffer_b, dtype=np.uint8)
        
        state_seg_tmp = np.concatenate((state_seg_tmp_f, state_seg_tmp_l), axis=0)
        state_seg_tmp = np.concatenate((state_seg_tmp, state_seg_tmp_r), axis=0)
        state_seg_tmp = np.concatenate((state_seg_tmp, state_seg_tmp_b), axis=0)

        state['seg'] = state_seg_tmp

        state['action'] = np.array(self.action_state_buffer, dtype=np.float32)

        if self.sp == 0:
            center_junc = carla.Location(x=-1.16, y=0.34, z=0.01)
            curr_pos = self.car.get_transform()
            if curr_pos.location.distance(center_junc) < 15:
                state['scene'] = 2
            else:
                state['scene'] = 0
        elif self.sp == 1 or self.sp == 8 or self.sp == 11 or self.sp == 12:
            center_junc = carla.Location(x=-1.16, y=0.34, z=0.01)
            curr_pos = self.car.get_transform()
            if curr_pos.location.distance(center_junc) < 15:
                state['scene'] = 1
            else:
                state['scene'] = 0
        elif self.sp == 2 or self.sp == 9:
            state['scene'] = 0
        elif self.sp == 3:
            state['scene'] = 0
        elif self.sp == 4 or self.sp == 10:
            state['scene'] = 0
        elif self.sp == 5:
            center_junc = carla.Location(x=-347.24, y=-366.67, z=0.01)
            curr_pos = self.car.get_transform()
            if curr_pos.location.distance(center_junc) < 15:
                state['scene'] = 2
            else:
                state['scene'] = 0
        elif self.sp == 6:
            center_junc = carla.Location(x=-347.24, y=-366.67, z=0.01)
            curr_pos = self.car.get_transform()
            if curr_pos.location.distance(center_junc) < 15:
                state['scene'] = 1
            else:
                state['scene'] = 0
        elif self.sp == 7:
            state['scene'] = 0

        return state
    
    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi/np.pi*180

    def pol2cart(self, rho, phi):
        phi = phi/180*np.pi
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y
    
    def step(self, action):
        self.timestep += 1

        action = copy.deepcopy(action)

        if action[0] >= -0.25 and action[0] <= 0.25:
            curr_steer = 0
            reward = 0
        elif action[0] > 0.25:
            curr_steer = (action[0]-0.25)/0.75
            reward = -0.4 * (curr_steer ** 1.4)
        elif action[0] < -0.25:
            curr_steer = (action[0]+0.25)/0.75
            curr_steer_reward = curr_steer*-1
            reward = -0.4 * (curr_steer_reward ** 1.4)

        self.curr_steer_position = self.curr_steer_position + (curr_steer * STEERING_SPEED)

        self.curr_steer_position = min(self.curr_steer_position, 0.8)
        self.curr_steer_position = max(self.curr_steer_position, -0.8)

        if self.curr_steer_position < 0.075 and self.curr_steer_position > -0.075:
            reward += 0.1

        control = carla.VehicleControl(throttle=MAX_THROTTLE, steer=self.curr_steer_position, brake=0,
                                       hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        for sub_step in range(2):
            if self.sp == 11:
                if self.timestep > 75 and self.timestep < 100:
                    control_obs = carla.VehicleControl(throttle=0.5, steer=0, brake=0, hand_brake=False,
                                                       reverse=False, manual_gear_shift=False, gear=0)
                else:
                    control_obs = carla.VehicleControl(throttle=0, steer=0, brake=1, hand_brake=False,
                                                       reverse=False, manual_gear_shift=False, gear=0)
                self.obs_car[self.which_obs].apply_control(control_obs)

            if self.sp == 12:
                if self.timestep > 75 and self.timestep < 85:
                    control_obs = carla.VehicleControl(throttle=0.5, steer=0, brake=0, hand_brake=False,
                                                       reverse=False, manual_gear_shift=False, gear=0)
                else:
                    control_obs = carla.VehicleControl(throttle=0, steer=0, brake=1, hand_brake=False,
                                                       reverse=False, manual_gear_shift=False, gear=0)
                self.obs_car[self.which_obs].apply_control(control_obs)

            self.car.apply_control(control)
            self.world.tick()

            while True:
                if self.seg_tmp_f is not None:
                    seg_state_f = self.seg_tmp_f
                    if sub_step == 1:
                        self.seg_state_buffer_f.append(seg_state_f)
                    self.seg_tmp_f = None
                    break

            while True:
                if self.seg_tmp_l is not None:
                    seg_state_l = self.seg_tmp_l
                    if sub_step == 1:
                        self.seg_state_buffer_l.append(seg_state_l)
                    self.seg_tmp_l = None
                    break

            while True:
                if self.seg_tmp_r is not None:
                    seg_state_r = self.seg_tmp_r
                    if sub_step == 1:
                        self.seg_state_buffer_r.append(seg_state_r)
                    self.seg_tmp_r = None
                    break

            while True:
                if self.seg_tmp_b is not None:
                    seg_state_b = self.seg_tmp_b
                    if sub_step == 1:
                        self.seg_state_buffer_b.append(seg_state_b)
                    self.seg_tmp_b = None
                    break

            if self.play:
                while True:
                    if self.rgb_tmp_f is not None:
                        rgb_state_f = self.rgb_tmp_f

                        self.play_images_seg[self.frame_play] = seg_state_f
                        self.play_images[self.frame_play] = rgb_state_f
                        self.frame_play += 1
                        break

            curr_pos = self.car.get_transform()
            x = round((curr_pos.location.x * self.m) + self.ref_point[1])
            y = round((curr_pos.location.y * self.m) + self.ref_point[0])

            if self.play:
                self.location_buffer.append([x, y, self.sp])
            
            yaw = -curr_pos.rotation.yaw
            img_car = np.ones((100, 100, 3), dtype=np.uint8)*255
            start_point = (round(1.43*self.m)+50, round(0.61*self.m)+50)
            end_point = (round(-1.73*self.m)+50, round(-0.61*self.m)+50)
            img_car = cv2.rectangle(img_car, start_point, end_point, (0, 0, 0), -1)
            img_car = ndimage.rotate(img_car, yaw, order=0, reshape=0, mode='nearest')

            img_car = cv2.bitwise_not(img_car[:, :, 0])/255
            img_road = self.reward_mask[self.use_this_scene[self.sp]][y-50:y+50, x-50:x+50, 0]/255
            try:
                bit_and = cv2.bitwise_and(img_car, img_road)
                dont_record = False
            except:
                dont_record = True
            
            if not dont_record:
                img_out_road = self.reward_mask[self.use_this_scene[self.sp]][y-50:y+50, x-50:x+50, 2]/255
                bit_out_road = cv2.bitwise_and(img_car, img_out_road)
                sum_all = np.sum(img_car)
                sum_and = np.sum(bit_and)
                sum_out = np.sum(bit_out_road)

                if self.sp in self.obs_location.keys():
                    curr_pos_obs = self.obs_car[self.which_obs].get_transform()

                    obs_range_mask = self.obs_class.curr_position(self.obstacles[self.which_obs][0], (curr_pos_obs.location.x, curr_pos_obs.location.y), curr_pos_obs.rotation.yaw)
                    obs_range_mask = obs_range_mask[y-50:y+50, x-50:x+50]/255
                    obs_bit_and = cv2.bitwise_and(img_car, obs_range_mask)
                    sum_mask = np.sum(obs_bit_and)

                velo = self.car.get_velocity()
                velo_r, velo_phi = self.cart2pol(velo.x, velo.y)
                convert_phi = yaw+velo_phi
                real_x, real_y = self.pol2cart(velo_r, convert_phi)
                real_x = real_x*3.6/8
                reward_real_x = (real_x-0.2)*3

                if self.sp in self.obs_location.keys():
                    if sum_mask >= 1:
                        reward -= 10
                        self.count_in_obs += 1
                    else:
                        reward += reward_real_x
                else:
                    reward += reward_real_x

                dis_end = []
                for n in range(len(self.end_points[self.sp])):
                    dis_end.append(curr_pos.location.distance(self.end_points[self.sp][n]))

                if curr_pos.location.distance(self.end_points[self.sp][2]) > self.prev_dist:
                    self.check_rev_dir.append(1)
                else:
                    self.check_rev_dir.append(0)

                self.prev_dist = curr_pos.location.distance(self.end_points[self.sp][2])

                if sum(self.check_rev_dir) >= N_CHECK_REVERSE:
                    end_this = True
                else:
                    end_this = False

                if sum_out > 0 or self.collision or end_this:
                    reward = -30
                    done = True
                    end = False

                elif min(dis_end) <= 2:
                    reward += ((sum_and/sum_all)-1)*1
                    if sum_and/sum_all < 0.98:
                        reward -= 1
                    done = True
                    end = True
                    
                else:
                    reward += ((sum_and/sum_all)-1)*1
                    if sum_and/sum_all < 0.98:
                        reward -= 1
                    done = False
                    end = False

                if done:
                    break

            else:
                break
        
        if not dont_record:
            return reward, done, end, dont_record
        else:
            return 0, True, False, dont_record

    def move_to_restpoint(self):
        self.car.set_transform(self.rest_points[0])

        if self.sp in self.obs_location.keys():
            self.obs_car[self.which_obs].set_transform(self.rest_points[self.which_obs+1])
