from environment.tools.scene_designer import locate_obstacle, create_point
from config.env import *
from config.camera import front_cam,left_cam,right_cam
from config.spawn_points import car_spawn
from utils.tools import carla_point
from environment.tools.action_wraper import five_action,dummy
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
import torch
from environment.tools.hud import HUD

ait_football_cw = reward_from_map("")

discrete_actions={1:[-1,0.5],2:[-0.12,0.6], 3:[0,0.8], 4:[0.12,0.6], 5:[1,0.5],0:[0,0]}
observation_space = spaces.Dict({
            'seg': spaces.Box(low=0, high=3, shape=(SEG_HIGHT,SEG_WIDTH,IN_CHANNLES), dtype=np.uint8)
        })
"""
[
                (carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
                (carla.Transform(carla.Location(x=2.5, y=0.5, z=0.0), carla.Rotation(pitch=-8.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
                (carla.Transform(carla.Location(x=0, y=-2.5, z=-0.0), carla.Rotation(yaw=90.0)), Attachment.Rigid)]
"""

class CarlaEnv(gym.Env):

    """
    open-ai environment for work with carla simulation 
    - connect with simulation 
    - initialize actor and sensor
    - send the image to 
    """


    def __init__(self,
                 host='localhost',
                 port=2000,
                 camera_list=[],
                 seg_model=None, 
                 vae_model=None,              
                 vehicle='evt_echo_4s',
                 car_spawn = (),
                 action_space_type = 'continuous',
                 observation_space = None,
                 delay = 0,
                 action_wraper = dummy(),
                 reward_fn = None,
                 verbose = 0,
                 seed=2024):
        
        
        if not len(camera_list):
            raise Exception("no sensor define")
        
        if not len(car_spawn):
            raise Exception("no car spawn point")
        
        if reward_fn==None:
            raise Exception("no reward func apply")
        
        # basic argument === 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        #make our work comparable 
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        
        self.seg_model = seg_model
        self.vae_model = vae_model
        self.action_space_type = action_space_type
        self.action_wraper = action_wraper
        self.reward_fn = reward_fn
        self.delay = delay
        self.render_display = verbose

        self.seg_state_buffer = deque(maxlen=IN_CHANNLES)
        self.action_state_buffer = deque(maxlen=IN_CHANNLES)

        # set gym action space ===
        if self.action_space_type == 'continuous':
            self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) 
            # self.observation_space['action'] = spaces.Box(low=)
             # steer = [-1,1] , throttle = [0,1]
        else:
            self.action_space = spaces.Discrete(len(discrete_actions))
            # self.observation_space['action'] = spaces.MultiDiscrete()
        # set gym observation space


        # connect to carla ===
        self.client = carla.Client(host, port)
        self.client.set_timeout(120)

        self.world = self.client.get_world()   
        self.set_world() 
        ## Destroy all actors
        self.world.tick()
        actors = self.world.get_actors()
        for actor in actors:
            actor.destroy()  

        # weather ==
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        self.blueprints = self.world.get_blueprint_library()
        self.bp_car = self.blueprints.filter(vehicle)[0]

        # list of spawn point ===
        self.carSPs = [carla_point(p) for p in car_spawn]
        
        self.car = self.world.spawn_actor(self.bp_car, self.carSPs[0])

        # set and camera parameter ===
        if seg_model == None:
            car_cam = 'sensor.camera.semantic_segmentation'
            preprocess = self.process_seg
        else:
            car_cam =  'sensor.camera.rgb'
            preprocess = self.process_rgb

        # attach it to car ===
        self.cams = []
        self.camera_list = camera_list
        self.camera_dict = {s['name']:s for s in self.camera_list}

        for s in self.camera_list:
            bp_cam = self.setting_camera(car_cam,s)
            cam = self.world.spawn_actor(bp_cam, carla.Transform(carla.Location(*s['Location']), carla.Rotation(*s['Rotation'])), attach_to=self.car)
            cam.listen(lambda data, cam_name =s['name'] : preprocess(data,cam_name))
            self.cams.append(cam)

        # cam for save video and visualize ===
        cam_3p = self.setting_camera('sensor.camera.rgb',s)
        self.cam_3p = self.world.spawn_actor(cam_3p, carla.Transform(carla.Location(x=-2.5, z=0.0), carla.Rotation(pitch=-8.0)), attach_to=self.car,attachment_type= carla.AttachmentType.SpringArmGhost)
        self.cam_3p.listen(lambda data: self.process_rgb(data))

        # Collision sensor ===
        self.bp_colli = self.blueprints.find('sensor.other.collision')
        self.colli_sensor = self.world.spawn_actor(self.bp_colli, carla.Transform(), attach_to=self.car)
        self.colli_sensor.listen(self.collision_callback)

        # pygame display ==
        if self.render_display:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((1120, 560), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
            # self.hud = HUD(1120, 560)
            # self.hud.set_vehicle(self.car)
            # self.world.on_tick(self.hud.on_world_tick)

        self.reset()
                                                                                                                                                               
    def setting_camera(self,cam_type,cam_config):
            
        bp_cam = self.blueprints.find(cam_type)
        bp_cam.set_attribute('image_size_y', str(cam_config['image_size_y']))
        bp_cam.set_attribute('image_size_x', str(cam_config['image_size_x']))
        bp_cam.set_attribute('fov', str(cam_config['fov']))
        bp_cam.set_attribute('sensor_tick', str(cam_config['sensor_tick']))

        return bp_cam
    

    def reset(self):

        # initial basic param ===

        self.curr_steer_position = 0
        self.count_in_obs = 0 # Step inside obstacle range
        self.collision = False # Check if hit an obstacle

        self.set_world()

        # telepot the car
        self.car.set_transform(self.carSPs[0])

        # set tmp of every camera to None
        self.cam_tmp = {s['name']:None for s in self.camera_list}


        self.world.tick()

        # spawn obstacle===


        # Set previous position for checking reverse===
        curr_pos = self.car.get_transform()
        self.prev_dist = curr_pos.location.distance(self.end_points[self.sp][2])
        
        # initial observation ===
        images = self.get_images()

        # fill image buffer with initial images
        for cn in self.seg_state_buffer.keys():
                
            self.seg_state_buffer[cn].extend([images[cn]]*IN_CHANNLES)
        
        # fill stay still action
        if self.action_space_type == "continuous":
            self.action_state_buffer.extend([[0,0]]*N_LOOK_BACK)
        else:
            self.action_state_buffer.extend([0]*N_LOOK_BACK)
        
        # record image===
            
        if self.vae_model != None:

            return self.get_latent_state()
        
        else:
            
            return self.get_state()


    def get_images(self):
        
        """
        return : {'cam',image}
        """

        states ={}
        for cam_name in self.cam_tmp.keys():
            while self.cam_tmp[cam_name] is None:
                pass
            image = self.cam_tmp[cam_name].copy()
            self.cam_tmp[cam_name] = None
            states[cam_name] = image

        return states
    
    def get_latent_state(self):

        for k,v in self.seg_state_buffer.items():


            self.vae_model(v)



    def get_state(self):

        """
        pack the image and action history together

        return : {'seg':{'cam':[image],} ,'action':[]}
        """

        state = {}
        seg_state_np = {k:np.array(v, dtype=np.uint8) for k,v in self.seg_state_buffer.items()}
        state['seg'] = seg_state_np
        state['action'] = np.array(self.action_state_buffer, dtype=np.float32)

        return state
    

    def step(self, action):

        # action = copy.deepcopy(action)

        if self.action_space_type == "continuous":
            steer, throttle = self.action_wraper(action=action,curent_steer = self.curr_steer_position)
        else:
            steer, throttle = discrete_actions[action]

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0,
                                       hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        self.car.apply_control(control)

        # ackermanncontrol ===
        # control = carla.VehicleAckermannControl(steer=steer, steer_speed=0.3 ,speed=throttle, acceleration=0.3, jerk=0.1)
        # self.car.apply_ackermann_control(control)
        
        # set obstable movement===

        self.world.tick()

        # get image from camera
        image = self.get_images()

        # get reward
        reward = self.reward_fn()


        # i don't know if this ok

        
        # if not dont_record:
        #     return reward, done, end, dont_record
        # else:
        #     return 0, True, False, dont_record
        
        return reward, done, end, dont_record
        

    
    def process_seg(self, data,cam_name):
        img = np.array(data.raw_data)
        img = img.reshape((self.camera_dict[cam_name]['image_size_y'], self.camera_dict[cam_name]['image_size_x'], 4))
        img = img[:, :, 2]
        seg_tmp = np.zeros([self.camera_dict[cam_name]['image_size_y'],self.camera_dict[cam_name]['image_size_x']], dtype=np.uint8)

        seg_tmp[img==1] = 1 # Road
        seg_tmp[img==24] = 2 # RoadLines
        seg_tmp[img==12] = 3 # Pedestrians
        seg_tmp[img==13] = 3 # Rider
        seg_tmp[img==14] = 3 # Car
        seg_tmp[img==15] = 3 # Truck
        seg_tmp[img==16] = 3 # Bus
        seg_tmp[img==18] = 3 # Motorcycle
        seg_tmp[img==19] = 3 # Bicycle

        self.cam_tmp[cam_name] = cv2.resize(seg_tmp, (self.camera_dict[cam_name]['image_size_y'], self.camera_dict[cam_name]['image_size_y']), interpolation=cv2.INTER_NEAREST)

    def process_rgb(self, data,cam_name):
        img = np.array(data.raw_data)
        img = img.reshape((self.camera_dict[cam_name]['image_size_y'], self.camera_dict[cam_name]['image_size_x'], 4))
        self.cam_tmp[cam_name] = img[:, :, 0:3].astype(np.uint8)

    def collision_callback(self, event):
        if event.other_actor.semantic_tags[0] not in self.check_seman_tags:
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

