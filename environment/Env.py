from environment.tools.UTILS import locate_obstacle, create_point
from config.env import env_config
from config.camera import front_cam,spectator_cam
from config.position import car_spawn
from utils.tools import carla_point
from environment.action_wraper import five_action,dummy
from environment.reward_fn import reward_from_map
import carla
import cv2
from collections import deque
import random
import numpy as np
import copy
from scipy import ndimage
import gym
from gym import spaces
import random
import pygame
import torch
# from environment.tools.hud import HUD
import warnings
import time


ait_football_cw = reward_from_map("")

class CarlaImageEnv(gym.Env):

    """
    open-ai environment for work with carla simulation server
    the function include 
    - send list of image (np array) from camera to observer return the state from observer
    - send the vehicle to reward_fn return the reward from reward_fn
    - return done if vehicle on destination or collis or out from road or go wrong direction
    - return info ??
    - send the command for control the car
    - construct the image to render in pygame

    """


    def __init__(self,
                 car_spawn = (),
                 discrete_actions = None,
                 observer = None,
                 delta_frame = 0.2,
                 action_wraper = dummy(),
                 reward_fn = None,    
                 env_config =env_config,
                 cam_config_list=[front_cam], 
                 activate_render = False,
                 seed=2024):
        
        
        if not len(cam_config_list):
            warnings.warn("no sensor config define")
        
        if not len(car_spawn):
            raise Exception("no car spawn point")
        
        if observer is None:
            raise Exception("no observer object apply")
        
        if reward_fn is None:
            raise Exception("no reward func apply")
        
        if not isinstance(discrete_actions,dict):
            raise Exception("discrete action have to be dict type")
        
        # make our work comparable 
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)

        # check if gpu available
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Select the default CUDA device
            properties = torch.cuda.get_device_properties(device)
            print("training on ", properties.name)
        else:
            device = torch.device("cpu")
            print("using cpu")

        # param ======================================================================================

        self.observer = observer
        self.action_wraper = action_wraper
        self.reward_fn = reward_fn
        self.discrete_actions = discrete_actions
        self.delta_frame = delta_frame
        self.activate_render = activate_render
        host = env_config.host
        port = env_config.port
        vehicle = env_config.vehicle
        self.max_step = env_config.max_step
        self.change_ep = env_config.change_point_ep
        self.current_point = 0
        self.ep = 0

        # set gym space ==============================================================================

        if discrete_actions == None :
            print("using continuous space steer = [-1,1] , throttle = [0,1]")
            self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) 
            
        else:
            print("using discret action")
            self.action_space = spaces.Discrete(len(discrete_actions))
            # self.observation_space['action'] = spaces.MultiDiscrete()

        self.observation_space = self.observer.gym_obs()
            
        # setting the carla ============================================================================

        self.client = carla.Client(host, port)
        self.client.set_timeout(120)
        self.world = self.client.get_world()  

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.delta_frame
        settings.synchronous_mode = True
        settings.max_substeps = 16
        settings.max_substep_delta_time = 0.0125
        self.world.apply_settings(settings)
        self.client.reload_world(False)

        # set weather 
        # self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # # Destroy all actors if there any 
        # self.world.tick()
        # actors = self.world.get_actors()
        # for actor in actors:
        #     actor.destroy()  

        # spawn car ===
        self.blueprints = self.world.get_blueprint_library()
        self.bp_car = self.blueprints.filter(vehicle)[0]
        # list of spawn point ===
        self.car_spawnponts = [carla_point(p) for p in car_spawn]        
        self.car = self.world.spawn_actor(self.bp_car, self.car_spawnponts[0])

        # attach cam to car ===
        self.cams = []
        self.cam_config_list = cam_config_list
        self.camera_dict = {s['name']:s for s in self.cam_config_list}

        for c in self.cam_config_list:

            if c.type == 'sensor.camera.semantic_segmentation':
                preprocess = self.process_seg
            elif c.type == 'sensor.camera.rgb':              
                preprocess = self.process_rgb
            else: 
                raise Exception(f"{c.type} not support yet")
            
            bp_cam = self.setting_camera(c)
            cam = self.world.spawn_actor(bp_cam, carla.Transform(carla.Location(*c.Location), carla.Rotation(*c.Rotation)), attach_to=self.car)
            cam.listen(lambda data, cam_name =c.name : preprocess(data,cam_name))
            self.cams.append(cam)

        # Collision sensor ===
        self.bp_colli = self.blueprints.find('sensor.other.collision')
        self.colli_sensor = self.world.spawn_actor(self.bp_colli, carla.Transform(), attach_to=self.car)
        self.colli_sensor.listen(self.collision_callback)

        if self.activate_render:
            # cam for save video and visualize ===
            Attachment = carla.AttachmentType
            self.spectator_rig = [
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]
            spec_cam = self.setting_camera('sensor.camera.rgb')
            self.spec_cam = self.world.spawn_actor(spec_cam, self.spectator_rig[0][0], attach_to=self.car,attachment_type= self.spectator_rig[0][1])
            self.spec_cam.listen(lambda data: self.process_rgb(data))

            # pygame display ==
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((640, 360), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
            # self.hud = HUD(1120, 560)
            # self.hud.set_vehicle(self.car)
                                                                                                                                                               
    def setting_camera(self,cam_config):
            
        bp_cam = self.blueprints.find(cam_config.type)
        for ak,av in cam_config.attribute.items():
            bp_cam.set_attribute(ak, str(av))

        return bp_cam
    

    def reset(self):

        # initial basic param ===============================================================
        self.curr_steer_position = 0
        self.count_in_obs = 0 # Step inside obstacle range
        self.collision = False
        self.ep+=1
        self.step_count = 0
        self.select_point()
        self.reset_car()

        # set tmp of every camera to None=====================================================
        self.cam_tmp = {s['name']:None for s in self.cam_config_list}


        self.world.tick()

        # spawn obstacle===


        # Set previous position for checking reverse===
        curr_pos = self.car.get_transform()
        self.prev_dist = curr_pos.location.distance(self.end_points[self.sp][2])

        # get the initial observation
        images = self.get_images()
        obs = self.observer.reset(images)

        return obs
    
    def select_point(self):
        """
        if it reach another points or reach change points ep it will change start point to the next point
        """
        if self.ep % self.change_ep ==0:
            self.current_point=self.current_point+1

        self.st = self.current_point%len(self.car_spawnponts)
        self.des = (self.current_point+1)%len(self.car_spawnponts)


    def reset_car(self):
        """
        teleport the car 
        """
        self.car.set_simulate_physics(False)
        self.car.set_transform(self.st)
        time.sleep(0.2)
        self.car.set_simulate_physics(True)

    def get_images(self):
        
        """
        return : list of images
        """

        images =[]
        for cam_name in self.cam_tmp.keys():
            while self.cam_tmp[cam_name] is None:
                pass
            image = self.cam_tmp[cam_name].copy()
            self.cam_tmp[cam_name] = None
            images.append(image)

        return images
    

    def step(self, action):

        # ackermanncontrol ===
        # control = carla.VehicleAckermannControl(steer=steer, steer_speed=0.3 ,speed=throttle, acceleration=0.3, jerk=0.1)
        # self.car.apply_ackermann_control(control)
        # set obstable movement===
        # action = copy.deepcopy(action)

        

        if self.discrete_actions == None:
            steer, throttle = self.action_wraper(action=action,curent_steer = self.curr_steer_position)
        else:
            steer, throttle = self.discrete_actions[action]

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0,
                                       hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        self.car.apply_control(control)

        self.world.tick()

        # get image from camera
        images = self.get_images()
        obs = self.observer.step(imgs = images,act=action)

        # get reward
        reward = self.reward_fn(self.car)

        # basic termination -> colision or reach max step or out of the rount more than n step 
        self.step_count+=1
        done = self.collision or self.step_count > self.max_step

        # get info
        info = {}
        
        return  obs,reward,done,info
     
    
    def process_seg(self, data,cam_name):
        img = np.array(data.raw_data)
        img = img.reshape((self.camera_dict[cam_name]['attribute']['image_size_y'], self.camera_dict[cam_name]['attribute']['image_size_x'], 4))
        img = img[:, :, 2]
        seg_tmp = np.zeros([self.camera_dict[cam_name]['attribute']['image_size_y'],self.camera_dict[cam_name]['attribute']['image_size_x']], dtype=np.uint8)

        seg_tmp[img==1] = 1 # Road
        seg_tmp[img==24] = 2 # RoadLines
        seg_tmp[img==12] = 3 # Pedestrians
        seg_tmp[img==13] = 3 # Rider
        seg_tmp[img==14] = 3 # Car
        seg_tmp[img==15] = 3 # Truck
        seg_tmp[img==16] = 3 # Bus
        seg_tmp[img==18] = 3 # Motorcycle
        seg_tmp[img==19] = 3 # Bicycle

        self.cam_tmp[cam_name] = cv2.resize(seg_tmp, (self.camera_dict[cam_name]['attribute']['image_size_y'], self.camera_dict[cam_name]['attribute']['image_size_y']), interpolation=cv2.INTER_NEAREST)

    def process_rgb(self, data,cam_name):
        img = np.array(data.raw_data)
        img = img.reshape((self.camera_dict[cam_name]['attribute']['image_size_y'], self.camera_dict[cam_name]['attribute']['image_size_x'], 4))
        self.cam_tmp[cam_name] = img[:, :, 0:3].astype(np.uint8)

    def collision_callback(self, event):
        if event.other_actor.semantic_tags[0] not in [1, 24]:
            self.collision = True

    def set_world(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.2
        settings.max_substeps = 16
        settings.max_substep_delta_time = 0.0125
        self.world.apply_settings(settings)

    def move_to_restpoint(self):
        self.car.set_transform(self.rest_points[0])

        if self.sp in self.obs_location.keys():
            self.obs_car[self.which_obs].set_transform(self.rest_points[self.which_obs+1])

    def close(self):
        pygame.quit()

    def render(self, mode="human"):
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

        # Tick render clock
        self.clock.tick()
        # self.hud.tick(self.world, self.clock)


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

    def help(self):
        print("""

            in order to render you need to give the spactator config 

            """)

