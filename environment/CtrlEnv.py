
from config.env.env_config import env_config_base
from config.env.camera import front_cam,spectator_cam
from environment.tools.action_wraper import OriginAction
from environment.tools.hud import get_actor_display_name
from environment.tools.actor_wrapper import *
from environment.tools.controllor import PygameManual
from environment.tools.scene_designer import *
import carla
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import weakref
import cv2
from environment.CarlaEnv import CarlaImageEnv

# sensor_transforms = {
#     "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
#     "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7)),}

class ManualCtrlEnv(CarlaImageEnv):

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
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }



    def __init__(self,
                 observer,
                 coach_config,
                 action_wrapper = OriginAction(), 
                 env_setting =dict(**env_config_base,max_step=1000),
                 cam_config_list=[front_cam], 
                 discrete_actions = None,
                 augment_image=False,
                 rand_weather=False,                 
                 seed=2024):
        activate_render = True,
        render_raw = True,
        render_observer = True,
        super().__init__(observer=observer,
                 coach_config=coach_config,
                 action_wrapper = action_wrapper, 
                 env_setting =env_setting,
                 cam_config_list=cam_config_list, 
                 discrete_actions = discrete_actions,
                 activate_render = activate_render,
                 render_raw = render_raw,
                 render_observer = render_observer,
                 augment_image=augment_image,
                 rand_weather=rand_weather,                 
                 seed=seed)

    def get_pygamecontroller(self):
        weak_self = weakref.ref(self)
        return PygameManual(spectator_cam,weak_self,self.discrete_actions)   
                                                                                                                                                  
    def reset(self, *, seed=None, options=None):
        self.reason = ""
        self.action = (0,0)
        obs,_ =  super().reset(seed=seed, options=options)
        return obs
    
    def step(self):
        # action = copy.deepcopy(action)
        if self.closed_env:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        # update param 
        self.step_count+=1

        action_command = self.pygamectrl.receive_key()
        if action_command is not None:
            self.action = action_command
            
        self.steer,self.throttle = self.action

        control = carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=False,
                                       hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        self.coach.set_movement()
        # control = carla.VehicleAckermannControl(steer=steer, steer_speed=0.3 ,speed=throttle, acceleration=3.6, jerk=0.1)
  

        self.world.tick()
        self.update_infos()

        self.car.apply_control(control)
        # self.car.apply_ackermann_control(control)

        # coach eval
        self.maneuver,self.reward,terminate,self.note = self.coach.review()
   
        self.total_reward+=self.reward

        # get image from camera
        self.list_images = self.world.get_all_obs()
        self.latent = self.observer.step(imgs = self.list_images,act=self.action)
        self.obs = np.concatenate((self.latent,self.maneuver))     

        # basic termination -> colision or reach max step or out of the rount more than n step 
        done = self.colli_sensor.collision or self.step_count > self.max_step or self.closed_env or terminate

        # get info
        info = {"step":self.step_count,
                "location":f"({self.car_position[0]:.2f},{self.car_position[1]:.2f})",
                "reward":self.reward,
                "total_reward":self.total_reward,
                "distance":self.distance,
                "total_distance":self.total_distance,
                "speed":self.speed,
                "avg_speed":self.avg_speed,
                "steer":self.steer,
                "mean_reward":self.mean_reward,
                **self.note}

        self.spec_image = self.spectator.get_obs()
        self.render()
    
        return  self.obs,self.reward,done,info


    def get_raw_images(self):
        return self.list_images
    
    def get_spectator_image(self,hud=True):
        if hud:
            return self.pygamectrl.get_display_array()
        else:
            return self.spec_image
        
    def get_input_states(self):
        return self.obs
