
from config.env.carla_sim import carla_setting
from config.env.camera import front_cam,spectator_cam
from environment.modules.action_wrapper import ActionBase
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
    description:
        this class is custom openai gym environment for work with carla environment

    args:
        observer(object): class that receive image and convert it to desired state
        coach_config(dict): config for manipulate the spawn points and object in the environments
        action_wrapper(object): class for post process the action
        carla_setting(dict): config for host,port,agent vehicle,frame duration
        agent_model(str): car model of agent
        max_step(int): limited time step of each episode
        cam_config_list(list): list of config of camera for spawn camera sensor
        discrete_actions(list): change the mode to discrete action([[steer,throttle]...]) this will deactivate the action_wrapper
        activate_render(bool): render the image while training
        render_raw(bool): render the raw images from camera
        render_observer(bool): render the images from observer
        augment_image(bool): activate image augmentation
        rand_weather(bool): random weather
        seed(int): set the random seed

    """

    def __init__(self,
                 observer,
                 coach_config,
                 action_wrapper = ActionBase(), 
                 carla_setting =carla_setting,
                 agent_model = 'evt_echo_4s',
                 max_step=1000,
                 cam_config_list=[front_cam], 
                 discrete_actions = None,
                 activate_render = False,
                 render_raw = False,
                 render_observer = False,
                 augment_image=False,
                 rand_weather=False,   
                 rand_cams_pos=False,              
                 seed=2024):
        
        if discrete_actions is not None:
            if not isinstance(discrete_actions,dict) and not isinstance(discrete_actions,list):
                raise Exception("discrete action have to be dict type or list type")
        
        random.seed(seed)
        # param ======================================================================================
        
        self.observer = observer
        self.action_wraper = action_wrapper
        self.discrete_actions = discrete_actions
        self.activate_render = activate_render
        self.render_raw = render_raw
        self.render_observer  = render_observer 
        self.rand_weather = rand_weather
        self.rand_cams_pos=rand_cams_pos

        self.max_step = max_step
        self.fps = 1/carla_setting['delta_frame']

        self.closed_env = False
        self.episode_idx =0


        # set gym space ==============================================================================

        if discrete_actions is None :
            print("using continuous space steer = [-1,1] , throttle = [0,1]")
            self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) 
            
        else:
            print("using discrete action")
            self.num_discrete = len(discrete_actions)
            self.action_space = spaces.Discrete(self.num_discrete)

        print(f"action space shape: {self.action_space.shape}")

    
        self.observation_space = observer.get_gym_space()
        print(f"observation space shape: {self.observation_space.shape}")

            
        self.world = World(carla_setting['host'],carla_setting['port'],carla_setting['delta_frame'])
        self.spawn_points = self.world.get_map().get_spawn_points()

        try:
            # create actor
            self.car = VehicleActor(self.world,
                                    agent_model,
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

    def eval_mode(self):
        print("setting coach to eval mode it will not randomly pick the scene, but select each scene in order ")  
        self.coach.eval_mode()                                                                                                                                                 
   
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
        self.steer = 0
        self.throttle = 0
        # reset actor, and 
        if self.rand_cams_pos:
            for cam in self.dcam:
                self.randomize_camera_transform(cam)
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
        self.obs = self.observer.reset(self.list_images)
        # print(self.latent,self.maneuver)
        if self.activate_render:
            self.spec_image = self.spectator.get_obs()
            self.render()
            self.pygamectrl.receive_key()

        return self.obs,{}
    
    def randomize_camera_transform(self,cam):
        # Random pitch, yaw, roll within the given ranges

        Location = list(cam.cam_config['Location'])
        Rotation = list(cam.cam_config['Rotation'])

        Rotation[0]+= random.uniform(-3, 3)
        Rotation[1]+= random.uniform(-3, 3)
        Rotation[2]+= random.uniform(-3, 3)

        # Random location with a small variation around [0.98, 0, 1.675] by +- 0.025
        Location[0]+= random.uniform(-0.025, 0.025)
        Location[1]+= random.uniform(-0.01, 0.01)
        Location[2]+= random.uniform(-0.025, 0.025)

        cam.set_transform(carla.Transform(carla.Location(*Location), carla.Rotation(*Rotation)))

    def step(self, action):


        # action = copy.deepcopy(action)
        if self.closed_env:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        # update param 
        self.step_count+=1

        # post process action
        if self.discrete_actions is None:
            self.steer, self.throttle,brake = self.action_wraper(action=action)
        else:
            self.steer, self.throttle = self.discrete_actions[action]
            brake = False
            # normalize to 0-1
            action = np.array([action/self.num_discrete])

        control = carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=brake,
                                       hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
        
        # set movement for other car ** not implement yet
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
        self.obs = self.observer.step(imgs = self.list_images,act=action,maneuver=self.maneuver)  

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

        if self.activate_render:
            self.spec_image = self.spectator.get_obs()
            self.render()
            self.pygamectrl.receive_key()
        
        return  self.obs,self.reward,done,self.closed_env,info
    
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
        

        list_of_strings = [f"{key}:{value}" for key, value in self.note.items()if key != "reason"]
        extra_info=[
            "Episode {}".format(self.episode_idx),
            "Step: {}".format(self.step_count),
            "",
            "Distance: % 7.3f m" % self.distance, 
            "Distance traveled: % 7d m" % self.total_distance,
            "speed:      % 7.2f km/h" % self.speed,
            f"steer value: {self.steer:.2f}",
            f"throttle value: {self.throttle:.2f}",
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

