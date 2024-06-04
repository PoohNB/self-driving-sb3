
from config.env.env_config import ait_football_env
from config.env.camera import front_cam,spectator_cam
from environment.tools.action_wraper import OriginAction
from environment.tools.hud import get_actor_display_name
import carla
import random
import numpy as np
import gym
from gym import spaces
import random
# from environment.tools.hud import HUD
from environment.tools.actor_wrapper import *
from environment.tools.controllor import PygameControllor
from environment.tools.scene_designer import *
import weakref
import cv2

# sensor_transforms = {
#     "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
#     "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7)),}

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
                 spawn_mode="static",
                 action_wrapper = OriginAction(), 
                 env_config =ait_football_env,
                 cam_config_list=[front_cam], 
                 discrete_actions = None,
                 activate_render = False,
                 render_raw = False,
                 render_seg = False,
                 render_reconst=False,
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
        self.render_raw = render_raw
        self.render_seg = render_seg
        self.render_reconst= render_reconst

        self.env_config = env_config
        self.max_step = env_config['max_step']
        self.env_config = env_config

        self.manual_end = False
        self.episode_idx =0

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
        self.car = VehicleActor(self.world,
                          env_config['vehicle'],
                          spawn_points=car_spawn)
        
        self.car.apply_mode(spawn_mode)
        
        self.rewarder.apply_car(self.car)
        
        # dash cam ===
        self.dcam = []
        for cf in cam_config_list:
            if cf["type"] =="sensor.camera.rgb":
                self.dcam.append(RGBCamera(self.world,self.car,cf))
            elif cf["type"] == "sensor.camera.semantic_segmentation":
                self.dcam.append(SegCamera(self.world,self.car,cf))

        # Collision sensor ===
        self.colli_sensor = CollisionSensor(self.world,self.car)
        self.rewarder.apply_collision_sensor(self.colli_sensor)

        if self.activate_render:
            # cam for save video and visualize ===
            self.spectator = SpectatorCamera(self.world,self.car,spectator_cam)
            # pygame display ==
            weak_self = weakref.ref(self)
            self.pygamectrl = PygameControllor(spectator_cam,weak_self)                                                                                                                                                              
   
    def reset(self):

        if self.manual_end:
            raise Exception("CarlaEnv.reset() called after the environment was closed.")
        # initial basic param ===============================================================
        self.episode_idx+=1
        self.curr_steer_position = 0
        self.count_in_obs = 0 # Step inside obstacle range
        self.step_count = 0
        self.total_reward = 0
        # reset actor   
        self.rewarder.reset()
        self.world.reset_actors() 
        # spawn obstacle===
        self.world.tick()
        # get the initial observation ========================================================
        self.list_images = self.world.get_all_obs()
        obs = self.observer.reset(self.list_images)

        return obs   

    def step(self, action):

        # update param 
    
        self.step_count+=1
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
        self.list_images = self.world.get_all_obs()
        obs = self.observer.step(imgs = self.list_images,act=action)
            
        # get reward
        
        self.reward = self.rewarder.reward(being_obstructed=False)
        self.total_reward+=self.reward

        # basic termination -> colision or reach max step or out of the rount more than n step 
        done = self.colli_sensor.collision or self.step_count > self.max_step or self.manual_end #or self.rewarder.get_terminate()

        # get info
        info = {}

        if self.activate_render:
            self.render()
            self.pygamectrl.receive_key()
        
        return  obs,self.reward,done,info
     

    def render(self):

        self.spec_image = self.spectator.get_obs()

        obs_list = []
        if self.render_raw:
            obs_list.append(self.list_images)
            
        if self.render_seg:
            obs_list.append(self.observer.get_seg_results())

        if self.render_reconst:
            reconstructed = self.observer.get_reconstructed()
            if reconstructed is None:
                self.render_reconst = False
                print("there no decoder in observer object")
            else:
                obs_list.append(reconstructed)

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

            

        extra_info=[
            "Episode {}".format(self.episode_idx),
            "Step: {}".format(self.step_count),
            "Reward: % 19.2f" % self.reward,
            "",
            "Distance traveled: % 7d m" % self.car.calculate_distance(),
            "speed:      % 7.2f km/h" % (self.car.get_velocity().length()),
            "Total reward:        % 7.2f" % self.total_reward,
        ]
        if self.colli_sensor.event is not None:
            self.pygamectrl.hud.notification("Collision with {}".format(get_actor_display_name(self.colli_sensor.event.other_actor)))
            self.colli_sensor.event=None
        self.pygamectrl.render(self.spec_image,extra_info) 

    # def attach_obs(self,obs_list):
    #     self.spec_image

    def close(self):
        if self.activate_render:
            self.pygamectrl.close()

        self.world.reset()

