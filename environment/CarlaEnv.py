
from config.env.env_config import env_config_base
from config.env.camera import front_cam,spectator_cam
from environment.tools.action_wraper import OriginAction
from environment.tools.hud import get_actor_display_name
from environment.tools.actor_wrapper import *
from environment.tools.controllor import PygameControllor
from environment.tools.scene_designer import *
from environment.tools.coach import Coach
from environment.tools.utils import calculate_distance
import carla
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import weakref
import cv2


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
                 activate_render = False,
                 render_raw = False,
                 render_observer = False,
                 augment_image=False,
                 rand_weather=False,                 
                 seed=2024):
        
        if discrete_actions is not None:
            if not isinstance(discrete_actions,dict) and not isinstance(discrete_actions,list):
                raise Exception("discrete action have to be dict type or list type")
        
        random.seed(seed)
        # param ======================================================================================
        
        self.maneuvers_set = {-1:'left',0:'forword',1:'right'}
        self.observer = observer
        self.action_wraper = action_wrapper
        self.discrete_actions = discrete_actions
        self.activate_render = activate_render
        self.render_raw = render_raw
        self.render_observer  = render_observer 
        self.rand_weather = rand_weather

        self.env_config = env_setting
        self.max_step = env_setting['max_step']
        self.fps = 1/env_setting['delta_frame']

        self.closed_env = False
        self.episode_idx =0

        # set gym space ==============================================================================

        if discrete_actions is None :
            print("using continuous space steer = [-1,1] , throttle = [0,1]")
            self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) 
            
        else:
            print("using discret action")
            self.num_discrete = len(discrete_actions)
            self.action_space = spaces.Discrete(self.num_discrete)


        num_maneuver = len(coach_config['scene_configs'][0]['cmd_config']['configs'][0]['cmd'])
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, self.observer.len_latent+num_maneuver),
                                                dtype=np.float32)
        
        print(f"observation space shape: {self.observation_space.shape}")
        print(f"action space shape: {self.action_space.shape}")
            
        self.world = World(env_setting['host'],env_setting['port'],env_setting['delta_frame'])
        self.spawn_points = self.world.get_map().get_spawn_points()

        try:
            # create actor
            self.car = VehicleActor(self.world,
                            env_setting['vehicle'],
                            self.spawn_points[0])
            
            # dash cam ===
            self.dcam = []
            for cf in cam_config_list:
                if cf["type"] =="sensor.camera.rgb":
                    self.dcam.append(RGBCamera(self.world,self.car,cf,augment_image))
                elif cf["type"] == "sensor.camera.semantic_segmentation":
                    self.dcam.append(SegCamera(self.world,self.car,cf,augment_image))

            # Collision sensor ===
            self.colli_sensor = CollisionSensor(self.world,self.car)
            weak_self = weakref.ref(self)
            self.coach = Coach(**coach_config,
                                      env=weak_self)

            if self.activate_render:
                # cam for save video and visualize ===
                self.spectator = SpectatorCamera(self.world,self.car,spectator_cam)
                # pygame display ==
                self.pygamectrl = self.get_pygamecontroller()
        except Exception as e:
            print(e)
            self.close()  

    def get_pygamecontroller(self):
        weak_self = weakref.ref(self)  
        return PygameControllor(spectator_cam,weak_self)                                                                                                                                                     
   
    def reset(self, *, seed=None, options=None):

        if self.closed_env:
            raise Exception("CarlaEnv.reset() called after the environment was closed.")
        # initial basic param ===============================================================
        self.episode_idx+=1
        # self.count_in_obs = 0 # Step inside obstacle range
        self.step_count = 0
        self.total_reward = 0
        self.total_speed = 0
        self.total_distance=0
        self.reward = 0
        self.prev_pos = None
        # reset actor, and 
        self.maneuver,self.note = self.coach.reset()
        self.world.reset_actors() 
        if self.rand_weather:
            self.world.random_wather()
        # advance step ====
        for _ in range(4):
            self.world.tick()
        self.update_infos()
        # get the initial observation ========================================================
        self.list_images = self.world.get_all_obs()
        self.latent = self.observer.reset(self.list_images)
        # print(self.latent,self.maneuver)
        self.obs = np.concatenate((self.latent,self.maneuver))
        if self.activate_render:
            self.spec_image = self.spectator.get_obs()
            self.render()
            self.pygamectrl.receive_key()

        return self.obs,{}

    def step(self, action):


        # action = copy.deepcopy(action)
        if self.closed_env:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        # update param 
        self.step_count+=1

        # post process action
        if self.discrete_actions is None:
            steer, throttle,brake = self.action_wraper(action=action)
        else:
            steer, throttle = self.discrete_actions[action]
            brake = False
            action = np.array([action/self.num_discrete])

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
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
        self.latent = self.observer.step(imgs = self.list_images,act=action)
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
                "steer":steer,
                "mean_reward":self.mean_reward,
                **self.note}

        if self.activate_render:
            self.spec_image = self.spectator.get_obs()
            self.render()
            self.pygamectrl.receive_key()
        
        return  self.obs,self.reward,done,False,info
    
    def get_raw_images(self):
        return self.list_images
    
    def get_spectator_image(self,hud=True):
        if hud:
            return self.pygamectrl.get_display_array()
        else:
            return self.spec_image
        
    def get_input_states(self):
        return self.obs
    
    def update_infos(self):
        self.car_position = self.car.get_xy_location()
        if self.prev_pos is None:
            self.prev_pos = self.car_position
        self.distance = calculate_distance((self.prev_pos),self.car_position)
        self.prev_pos= self.car_position
        self.total_distance += self.distance
        self.speed = self.car.get_xy_velocity()*3.6
        self.total_speed +=self.speed
        self.avg_speed = self.total_speed/self.step_count if self.step_count >0 else 0
        self.mean_reward = self.total_reward/self.step_count if self.step_count >0 else 0

    def render(self):
        
        obs_list = []
        if self.render_raw:
            obs_list.append(self.list_images)
            
        if self.render_observer:
            obs_list.extend(self.observer.get_renders())

        # Resize and arrange images
        spec_height, spec_width,_ = self.spec_image.shape
        target_height = spec_height // 6

        x_offset = spec_width

        for img_set in obs_list:
            resized_images = []
            total_height = 0

            # Resize images and calculate the total height
            for img in img_set:
                height, width, _ = img.shape
                scaling_factor = target_height / height
                new_width = int(width * scaling_factor)
                resized_img = cv2.resize(img, (new_width, target_height))
                resized_images.append(resized_img)
                total_height += target_height

            # Calculate the x offset for the current list
            x_offset -= new_width

            # Place images in the main image
            y_offset = 0
            for resized_img in resized_images:
                self.spec_image[y_offset:y_offset + target_height, x_offset:x_offset + new_width] = resized_img
                y_offset += target_height
        

        manv = self.maneuvers_set[self.maneuver[0]]
        list_of_strings = [f"{key}:{value}" for key, value in self.note.items()if key != "reason"]
        extra_info=[
            "Episode {}".format(self.episode_idx),
            "Step: {}".format(self.step_count),
            "",
            f"Maneuver: {manv}",
            "Distance: % 7.3f m" % self.distance, 
            "Distance traveled: % 7d m" % self.total_distance,
            "speed:      % 7.2f km/h" % self.speed,
            "Reward: % 19.2f" % self.reward,
            "Total reward:        % 7.2f" % self.total_reward,
        ]+list_of_strings

        if self.colli_sensor.event is not None:
            self.pygamectrl.hud.notification("Collision with {}".format(get_actor_display_name(self.colli_sensor.event.other_actor)))
            self.colli_sensor.event=None
  
        self.pygamectrl.render(self.spec_image,extra_info) 


    


    def close(self):
        if self.activate_render:
            try:
                self.pygamectrl.close()
            except:
                pass

        self.world.reset()
        self.closed_env = True

