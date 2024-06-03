from environment.tools.scene_designer import locate_obstacle, create_point
from config.env import env_config
from config.camera import front_cam
from utils.tools import carla_point
from environment.tools.action_wraper import action_dummy
from environment.tools.rewarder import reward_dummy
from environment.tools.rewarder import reward_from_map
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
from pygame.locals import K_ESCAPE,K_TAB
import torch
# from environment.tools.hud import HUD
import warnings
import time
from carla import ColorConverter as cc


sensor_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7)),}

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

    note for step: in real world after model receive image from camera it will use some time to predict the action (which is delta time between frame) 
    then send the action and receive the next image simutaneosly so the step is not predict-->apply action-->tick world, 
      but predict -->apply last action --> tick world or predict-->tick world-->apply function

    """


    def __init__(self,
                 car_spawn = (),
                 discrete_actions = None,
                 observer = None,
                 reward_fn = None,  
                 delta_frame = 0.2,
                 action_wraper = action_dummy,  
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
        
        if discrete_actions is not None and not isinstance(discrete_actions,dict):
            raise Exception("discrete action have to be dict type")
        
        random.seed(seed)

        # check if gpu available
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Select the default CUDA device
            properties = torch.cuda.get_device_properties(device)
            print("using ", properties.name)
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
        self.cam_config_list = cam_config_list
        self.camera_dict = {s['name']:s for s in self.cam_config_list}
        self.car_spawnponts = [carla_point(p) for p in car_spawn]   

        host = env_config['host']
        port = env_config['port']
        vehicle = env_config['vehicle']
        self.max_step = env_config['max_step']
        self.change_ep = env_config['change_point_ep']

        self.current_point = 0
        self.ep = 0
        self.spectator_index = 0
        self.reach_next_point = False
        self.render_obs_pic = False
        self.actor_list = []
        self.manual_end = False

        self.spectator_config =  dict(
                                type = "sensor.camera.rgb",
                                attribute= dict(
                                                image_size_x=1280,
                                                image_size_y=720
                                                )
                                )
        


        # set gym space ==============================================================================

        if discrete_actions == None :
            print("using continuous space steer = [-1,1] , throttle = [0,1]")
            self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) 
            
        else:
            print("using discret action")
            self.action_space = spaces.Discrete(len(discrete_actions))
            # self.observation_space['action'] = spaces.MultiDiscrete()

        self.observation_space = self.observer.gym_obs()
            
        # setting world ============================================================================

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
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Destroy all actors if there any 
        # self.world.tick()

        # create actor
        self.blueprints = self.world.get_blueprint_library()
        self.bp_car = self.blueprints.filter(vehicle)[0]
        # list of spawn point ===     
        self.car = self.world.spawn_actor(self.bp_car, self.car_spawnponts[0])

        # attach cam to car ===
        self._create_observer_cam()

        # Collision sensor ===
        self.bp_colli = self.blueprints.find('sensor.other.collision')
        self.colli_sensor = self.world.spawn_actor(self.bp_colli, carla.Transform(), attach_to=self.car)
        self.colli_sensor.listen(self.collision_callback)
        self.actor_list.append(self.colli_sensor)

        if self.activate_render:
            # cam for save video and visualize ===
            Attachment = carla.AttachmentType

            bound_x = 0.5 + self.car.bounding_box.extent.x
            bound_y = 0.5 + self.car.bounding_box.extent.y
            bound_z = 0.5 + self.car.bounding_box.extent.z
            self.spectator_rig = [
                (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
                # (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=0.0, y=0.0, z=+6*bound_z), carla.Rotation(pitch=-90.0)),Attachment.Rigid),
                # (carla.Transform(carla.Location(x=-400, y=-200, z=500.0), carla.Rotation(pitch=-90.0)), None)
                ]
            # self.spectator_rig.extend([(carla.Transform(carla.Location(*c['Location']),carla.Rotation(*c['Rotation'])), Attachment.Rigid) for c in self.cam_config_list])
            self.spec_cam_bp = self._setting_camera(self.spectator_config)
            self._create_spectator_cam()

            # pygame display ==
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((self.spectator_config['attribute']['image_size_x'], self.spectator_config['attribute']['image_size_y']), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
                                                                                                                                                               
    def _setting_camera(self,cam_config):
            
        bp_cam = self.blueprints.find(cam_config['type'])
        for ak,av in cam_config['attribute'].items():
            bp_cam.set_attribute(ak, str(av))

        return bp_cam
    
    def change_spectator_view(self):
        self.spectator_index=(self.spectator_index+1)%(len(self.spectator_rig)+len(self.cam_config_list))
        if self.spectator_index >= len(self.spectator_rig):
            self.render_obs_pic = True
        else:
            self.render_obs_pic = False
            self.spec_cam.destroy()
            self._create_spectator_cam()

    def _create_spectator_cam(self):

        self.spec_cam = self.world.spawn_actor(self.spec_cam_bp, self.spectator_rig[self.spectator_index][0], attach_to=self.car,attachment_type= self.spectator_rig[self.spectator_index ][1])
    
        self.spec_cam.listen(lambda data: self.process_spectator(data))
        

    def _create_observer_cam(self):
        # self.cams = []
        self.camera_dict = {s['name']:s for s in self.cam_config_list}
        for c in self.cam_config_list:

            if c['type'] == 'sensor.camera.semantic_segmentation':
                preprocess = self.process_seg
            elif c['type'] == 'sensor.camera.rgb':              
                preprocess = self.process_rgb
            else: 
                raise Exception(f"{c['type']} not support yet")
            
            bp_cam = self._setting_camera(c)
            cam = self.world.spawn_actor(bp_cam, carla.Transform(carla.Location(*c['Location']), carla.Rotation(*c['Rotation'])), attach_to=self.car)
            cam.listen(lambda data, cam_name =c['name'] : preprocess(data,cam_name))
            self.actor_list.append(cam)
     
   
    def reset(self):

        if self.manual_end:
            raise Exception("CarlaEnv.reset() called after the environment was closed.")

        # initial basic param ===============================================================
        self.curr_steer_position = 0
        self.count_in_obs = 0 # Step inside obstacle range
        self.collision = False
        self.ep+=1
        self.step_count = 0

        self.reset_car()

        # set tmp of every camera to None=====================================================
        self.cam_tmp = {s['name']:None for s in self.cam_config_list}

        if self.activate_render:

            self.spec_cam_tmp = None
       
        
        # spawn obstacle===

        self.world.tick()

        # get the initial observation ========================================================
        self.list_images = self.get_images()
        obs = self.observer.reset(self.list_images)


        return obs
    
    def select_point(self):
        """
        if it reach another points or reach change points ep it will change start point to the next point
        """
        if self.ep % self.change_ep ==0:
            self.current_point=self.current_point+1
        elif self.reach_next_point:
            self.current_point=self.current_point+1
            self.reach_next_point = False

        self.st = self.current_point%len(self.car_spawnponts)
        self.des = (self.current_point+1)%len(self.car_spawnponts)


    def reset_car(self):
        """
        teleport the car 
        """
        self.select_point()
        self.car.set_simulate_physics(False)
        self.car.set_transform(self.car_spawnponts[self.st])
        time.sleep(0.2)
        self.car.set_simulate_physics(True)

    def get_images(self):
        
        """
        return : list of images
        """

        images = []
        for cam_name in self.cam_tmp.keys():
            while self.cam_tmp[cam_name] is None:
                pass
            image = self.cam_tmp[cam_name].copy()
            self.cam_tmp[cam_name] = None
            images.append(image)

        return images
    
    def get_spectator_image(self):
        while self.spec_cam_tmp is None:
            pass
        image = self.spec_cam_tmp.copy()
        self.spec_cam_tmp = None

        return image
    

    def step(self, action):

        # ackermanncontrol ===
        # control = carla.VehicleAckermannControl(steer=steer, steer_speed=0.3 ,speed=throttle, acceleration=0.3, jerk=0.1)
        # self.car.apply_ackermann_control(control)
        # set obstable movement===
        # action = copy.deepcopy(action)
        if self.manual_end:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        if self.discrete_actions == None:
            steer, throttle = self.action_wraper(action=action,curent_steer = self.curr_steer_position)
        else:
            steer, throttle = self.discrete_actions[action]

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0,
                                       hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        self.car.apply_control(control)

        self.world.tick()

        # get image from camera
        self.list_images = self.get_images()
        obs = self.observer.step(imgs = self.list_images,act=action)

        if self.activate_render:
            self.render()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.close()
                        self.manual_end = True
                    elif event.key == pygame.K_TAB:
                        self.change_spectator_view()
            

        # get reward
        reward = self.reward_fn.reward(self.car)

        # basic termination -> colision or reach max step or out of the rount more than n step 
        self.step_count+=1
        done = self.collision or self.step_count > self.max_step or self.manual_end

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

        # data.convert(cc.CityScapesPalette)
        # seg_tmp = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        # seg_tmp = np.reshape(seg_tmp, (data.height, data.width, 4))
        # seg_tmp = seg_tmp[:, :, :3]
        # seg_tmp = seg_tmp[:, :, ::-1]
        
        # self.cam_tmp[cam_name] = seg_tmp

        
    def process_rgb(self, data,cam_name):
        img = np.array(data.raw_data)
        img = img.reshape((self.camera_dict[cam_name]['attribute']['image_size_y'], self.camera_dict[cam_name]['attribute']['image_size_x'], 4))
        self.cam_tmp[cam_name] = img[:, :, :3][:, :, ::-1].astype(np.uint8)#[:, :, ::-1]

    def process_spectator(self,data):
        img = np.array(data.raw_data)
        img = img.reshape((self.spectator_config['attribute']['image_size_y'], self.spectator_config['attribute']['image_size_x'], 4))
        self.spec_cam_tmp = img[:, :, :3][:, :, ::-1].astype(np.uint8)


    def collision_callback(self, event):
        if event.other_actor.semantic_tags[0] not in [1, 24]:
            self.collision = True

    def move_to_restpoint(self):
        self.car.set_transform(self.rest_points[0])

        if self.sp in self.obs_location.keys():
            self.obs_car[self.which_obs].set_transform(self.rest_points[self.which_obs+1])

    def close(self):
        if self.activate_render:
            pygame.quit()

        self.reset_world()

    def render(self):

        if self.render_obs_pic:
            
            self.spec_image = self.list_images[self.spectator_index-len(self.spectator_rig)].copy()
        else:
  
            self.spec_image = self.get_spectator_image()

        # Tick render clock
        self.clock.tick()
        # self.hud.tick(self.world, self.clock)

        self.display.blit(pygame.surfarray.make_surface(self.spec_image.swapaxes(0, 1)), (0, 0))

        pygame.display.flip()


    def help(self):
        print("""

            in order to render you need to give the spactator config 

            """)


    def reset_world(self):
        try:
            self.spec_cam.destroy()
            for actor in self.actor_list:

                    actor.destroy()

        except:
            print("sensor already destroy")  

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
        
        # self.world.apply_settings(self.ori_settings)



    # def set_world(self):
    #     settings = self.world.get_settings()
    #     settings.synchronous_mode = True
    #     settings.fixed_delta_seconds = 0.2
    #     settings.max_substeps = 16
    #     settings.max_substep_delta_time = 0.0125
    #     self.world.apply_settings(settings)



