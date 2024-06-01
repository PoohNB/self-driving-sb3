from environment.tools.scene_designer import locate_obstacle, create_point
from config.env.env_config import env_config
from config.env.camera import front_cam,spectator_cam
from utils.tools import carla_point
from environment.tools.action_wraper import OriginAction
from environment.tools.rewarder import reward_dummy
from environment.tools.rewarder import reward_from_map
from environment.tools.hud import get_actor_display_name
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
from environment.tools.actor_wrapper import *
from environment.tools.controllor import PygameControllor
from environment.tools.scene_designer import *

sensor_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7)),}

class CarlaImageEnv(gym.Env):

    """
    open-ai environment for work with carla simulation server

    the function include 
    - send list of image (np array) from camera to observer return the state from observer
    - send the vehicle to rewarder return the reward from rewarder
    - return done if vehicle on destination or collis or out from road or go wrong direction
    - return info ??
    - send the command for control the car
    - construct the image to render in pygame

    note for step: in real world after model receive image from camera it will use some time to predict the action (which is delta time between frame) 
    then send the action and receive the next image simutaneosly so the step is not predict-->apply action-->tick world, 
      but predict -->apply last action --> tick world or predict-->tick world-->apply function

    """


    def __init__(self,
                 observer,
                 rewarder,   
                 car_spawn,
                 action_wrapper = OriginAction(), 
                 env_config =env_config,
                 cam_config_list=[front_cam], 
                 discrete_actions = None,
                 activate_render = False,
                 seed=2024):
        
        if discrete_actions is not None and not isinstance(discrete_actions,dict):
            raise Exception("discrete action have to be dict type")
        
        random.seed(seed)
        # param ======================================================================================

        self.observer = observer
        self.action_wraper = action_wrapper
        self.rewarder = rewarder
        self.discrete_actions = discrete_actions
        self.activate_render = activate_render

        self.env_config = env_config
        self.max_step = env_config['max_step']
        self.env_config = env_config

        self.manual_end = False

        # set gym space ==============================================================================

        if discrete_actions == None :
            print("using continuous space steer = [-1,1] , throttle = [0,1]")
            self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) 
            
        else:
            print("using discret action")
            self.action_space = spaces.Discrete(len(discrete_actions))

        self.observation_space = self.observer.gym_obs()
            
        self.world = World(env_config['host'],env_config['port'],env_config['delta_frame'])

        # create actor
        self.car = Vehicle(self.world,
                          env_config['vehicle'],
                          spawn_points=car_spawn)
        
        # dash cam ===
        self.dcam = []
        for cf in cam_config_list:
            if cf["type"] =="sensor.camera.rgb":
                self.dcam.append(RGBCamera(self.world,self.car,cf))
            elif cf["type"] == "sensor.camera.semantic_segmentation":
                self.dcam.append(SegCamera(self.world,self.car,cf))

        # Collision sensor ===
        self.colli_sensor = CollisionSensor(self.world,self.car)

        if self.activate_render:
            # cam for save video and visualize ===
            self.spectator = SpectatorCamera(self.world,self.car,spectator_cam)
            # pygame display ==
            self.pygamectrl = PygameControllor(spectator_cam,self)                                                                                                                                                              
   
    def reset(self):

        if self.manual_end:
            raise Exception("CarlaEnv.reset() called after the environment was closed.")
        # initial basic param ===============================================================
        self.curr_steer_position = 0
        self.count_in_obs = 0 # Step inside obstacle range
        self.step_count = 0
        # reset actor   
        self.world.reset_actors() 
        # spawn obstacle===
        self.world.tick()
        # get the initial observation ========================================================
        self.list_images = self.world.get_obs()
        obs = self.observer.reset(self.list_images)

        return obs   

    def step(self, action):

        # set obstable movement===
        # action = copy.deepcopy(action)
        if self.manual_end:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        if self.discrete_actions == None:
            steer, throttle,brake = self.action_wraper(action=action)
        else:
            steer, throttle = self.discrete_actions[action]
            brake = 0

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                       hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        # control = carla.VehicleAckermannControl(steer=steer, steer_speed=0.3 ,speed=throttle, acceleration=0.3, jerk=0.1)
        # self.car.apply_ackermann_control(control)

        self.world.tick()

        self.car.apply_control(control)

        # get image from camera
        self.list_images = self.world.get_obs()
        obs = self.observer.step(imgs = self.list_images,act=action)

        if self.activate_render:
            self.render()
            self.pygamectrl.receive_key()
            
        # get reward
        reward = self.rewarder.reward(self.car)

        # basic termination -> colision or reach max step or out of the rount more than n step 
        self.step_count+=1
        done = self.collision or self.step_count > self.max_step or self.manual_end

        # get info
        info = {}
        
        return  obs,reward,done,info
     

    def render(self):

        self.spec_image = self.spectator.get_obs()

        if self.show_obs:
            pass

        self.pygamectrl.hud.notification("Collision with {}".format(get_actor_display_name(self.colli_sensor.event.other_actor)))
        self.pygamectrl.render(self.spec_image)   

    def close(self):
        if self.activate_render:
            self.pygamectrl.close()

        self.world.reset()

