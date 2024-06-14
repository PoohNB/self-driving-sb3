# ==============================================================================
# -- copied and modified from Alberto Maté Angulo --
# ==============================================================================


import threading
import carla
import numpy as np
import cv2
from carla import ColorConverter as cc
import time
import random
from typing import List
import weakref
import math

class CarlaActorBase:
    def __init__(self,
                  wrapped_world,
                  actor,
                  tag="unknown"):
        
        self.actor = actor
        self.destroyed = False
        self.tag = tag
        self.wrapped_world = wrapped_world
        self.append_to_world(self.wrapped_world,self.tag)

    def append_to_world(self,world,tag):
        if tag =="obs":
            world.append_observer(self)
        else:
            world.append(self)
        print(f"initialize {tag} actor")

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.destroyed = True

            if self in self.wrapped_world.actor_list:
                self.wrapped_world.actor_list.remove(self)

            if self in self.wrapped_world.observer_list:
                self.wrapped_world.observer_list.remove(self)

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)


class CameraBase(CarlaActorBase):

    def __init__(self,
                 wrapped_world,
                 wrapped_veh,
                 cam_config,
                 turbulence = False):
        
        # =============================================
        ## init param from super class manually
        # self.actor = actor
        # self.destroyed = False
        # self.tag = tag
        # self.wrapped_world = wrapped_world
        # =============================================
        self.lock = threading.Lock()
        self.cam_tmp = None
        self.tag = cam_config['tag']
        self.turb = turbulence
        self.world = wrapped_world.get_carla_world()
        self.wrapped_world = wrapped_world
        self.car = wrapped_veh.get_carla_actor()
        self.blueprints = self.world.get_blueprint_library()
        self.cam_config = cam_config
        self.bp_cam = self._setting_camera(cam_config)
        self.create(cam_config=cam_config)


    def create(self,cam_config):
        self.actor = self.world.spawn_actor(self.bp_cam, carla.Transform(carla.Location(*cam_config['Location']), carla.Rotation(*cam_config['Rotation'])), 
                                     attach_to=self.car,attachment_type=cam_config['AttachmentType'])
        weak_self = weakref.ref(self)
        self.actor.listen(lambda data : self.process(weak_self,data))
        self.destroyed = False
        self.append_to_world(self.wrapped_world,self.cam_config['tag'])
        

    def reset(self):
        self.cam_tmp =None

    def _setting_camera(self,cam_config):
            
        bp_cam = self.blueprints.find(cam_config['type'])
        for ak,av in cam_config['attribute'].items():
            bp_cam.set_attribute(ak, str(av))

        return bp_cam
   
    def get_obs(self):
        
        """
        return : list of images
        """

        assert not self.destroyed
        
        while self.cam_tmp is None:
            pass
        image = self.cam_tmp.copy()
        self.cam_tmp = None

        if self.turb:
            image = self.apply_turbulence(image)

        return image
    
    @staticmethod
    def apply_turbulence(image):
        rows, cols = image.shape[:2]
        
        # Random rotation angle between -10 and 10 degrees
        angle = random.uniform(-2, 2)
        M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        
        # Random shift values between -5 and 5 pixels
        tx = random.uniform(-5, 5)
        ty = random.uniform(-5, 5)
        M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
        # Apply rotation
        rotated_image = cv2.warpAffine(image, M_rotate, (cols, rows))
        # Apply shift
        transformed_image = cv2.warpAffine(rotated_image, M_shift, (cols, rows))
        
        return transformed_image


    def process(self,weak_self,data):
        self_ref = weak_self()
        if not self_ref:
            return

        raise NotImplementedError("Method 'process' must be implemented in subclasses")
    

class RGBCamera(CameraBase):
    """
    give rgb image
    """

    def __init__(self,
                 world,
                 car,
                 cam_config,
                 turbulence=False):
        
        assert cam_config['type'] == 'sensor.camera.rgb'
        
        super().__init__(world,
                         car,
                         cam_config,
                         turbulence)
        
    def process(self,weak_self,data):
        self_ref = weak_self()
        if not self_ref:
            return
        img = np.array(data.raw_data)
        img = img.reshape((self_ref.cam_config['attribute']['image_size_y'], self_ref.cam_config['attribute']['image_size_x'], 4))
        self_ref.cam_tmp = img[:, :, :3][:, :, ::-1].astype(np.uint8)#BGRA2RGB

#=====

custom_palette = {1:1,24:2,12:3,13:3,14:3,15:3,16:3,18:3,19:3}
# seg_tmp[img==1] # Road
# seg_tmp[img==24] # RoadLines
# seg_tmp[img==12] # Pedestrians
# seg_tmp[img==13] # Rider
# seg_tmp[img==14] # Car
# seg_tmp[img==15] # Truck
# seg_tmp[img==16] # Bus
# seg_tmp[img==18] # Motorcycle
# seg_tmp[img==19] # Bicycle

class SegCamera(CameraBase):

    def __init__(self,
                 world,
                 car,
                 cam_config,
                 palette = custom_palette,
                 turbulence=False):
        
        assert cam_config['type'] == 'sensor.camera.semantic_segmentation'
        
        self.palette = palette
        
        super().__init__(world,
                         car,
                         cam_config,
                         turbulence)
        
    def process(self,weak_self,data):
        self_ref = weak_self()
        if not self_ref:
            return

        if self_ref.palette == "CityScapesPalette":
            self_ref.process_CityScapesPalette(data)
        else:
            self_ref.process_seg(data)
    
    def process_seg(self, data):
        
        img = np.array(data.raw_data)
        img = img.reshape((self.cam_config['attribute']['image_size_y'], self.cam_config['attribute']['image_size_x'], 4))
        img = img[:, :, 2]
        seg_tmp = np.zeros([self.cam_config['attribute']['image_size_y'],self.cam_config['attribute']['image_size_x']], dtype=np.uint8)

        for k,v in self.palette.items():
            seg_tmp[img==k] = v

        self.cam_tmp = cv2.resize(seg_tmp, (self.cam_config['attribute']['image_size_y'], self.cam_config['attribute']['image_size_y']), interpolation=cv2.INTER_NEAREST)

    def process_CityScapesPalette(self,data):

        data.convert(cc.CityScapesPalette)
        seg_tmp = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        seg_tmp = np.reshape(seg_tmp, (data.height, data.width, 4))
        seg_tmp = seg_tmp[:, :, :3]
        seg_tmp = seg_tmp[:, :, ::-1]
        
        self.cam_tmp = seg_tmp

class SpectatorCamera(RGBCamera):


    def __init__(self,
                 wrapped_world,
                 wrapped_veh,
                 cam_config):
        
        car = wrapped_veh.get_carla_actor()
        Attachment = carla.AttachmentType
        bound_x = 0.5 + car.bounding_box.extent.x
        bound_y = 0.5 + car.bounding_box.extent.y
        bound_z = 0.5 + car.bounding_box.extent.z

        # ((x,y,z),(pitch,yaw,roll),attachment)
        self.perceptions = [
                dict(Location=(-2.0*bound_x, +0.0*bound_y, 2.0*bound_z), Rotation=(8.0,0,0), AttachmentType=Attachment.SpringArmGhost),
                dict(Location=(+1.9*bound_x, +1.0*bound_y, 1.2*bound_z),Rotation=(0,0,0) , AttachmentType=Attachment.SpringArmGhost),
                dict(Location=(-2.8*bound_x, +0.0*bound_y, 4.6*bound_z), Rotation=(6.0,0,0), AttachmentType=Attachment.SpringArmGhost),
                dict(Location=(-1.0, -1.0*bound_y, 0.4*bound_z),Rotation=(0,0,0) ,AttachmentType=Attachment.Rigid),
                dict(Location=(0.0, 0.0, +6*bound_z), Rotation=(-90.0,0,0),AttachmentType=Attachment.Rigid),
                ]
        self.spectator_index = 0
        cam_config['Location'] = self.perceptions[0]['Location']
        cam_config['Rotation'] = self.perceptions[0]['Rotation']
        cam_config['AttachmentType'] = self.perceptions[0]['AttachmentType']
        cam_config['tag'] = "spec"
        
        super().__init__(wrapped_world,
                         wrapped_veh,
                         cam_config)
        
    def change_perception(self):
        self.spectator_index=(self.spectator_index+1)%len(self.perceptions)
        self.destroy()
        self.create(self.perceptions[self.spectator_index])
 

        
class CollisionSensor(CarlaActorBase):

    def __init__(self,wrapped_world,wrapped_veh):
        self.world = wrapped_world.get_carla_world()
        self.car =  wrapped_veh.get_carla_actor()
        blueprints = self.world.get_blueprint_library()
        self.bp_colli = blueprints.find('sensor.other.collision')
        self.colli_sensor = self.world.spawn_actor(self.bp_colli, carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.colli_sensor.listen(lambda event: self.collision_callback(weak_self, event))
        super().__init__(wrapped_world,self.colli_sensor,tag="colli")

    def reset(self):
        self.collision = False
        self.event=None

    def collision_callback(self,weak_self,event):

        self_ref = weak_self()
        if not self_ref:
            return
        
        if event.other_actor.semantic_tags[0] not in [1, 24]:
            self_ref.collision = True
        self_ref.event = event

class VehicleActor(CarlaActorBase):

    def __init__(self,
                 wrapped_world,
                 vehicle_name,
                 spawn_points,
                 startpoint=0,
                 change_points_ep = 100,
                 mode = "static"):
        
        self.st = startpoint
        self.world = wrapped_world.get_carla_world()
        self.spawn_points = [carla.Transform(carla.Location(*sp['Location']), carla.Rotation(*sp['Rotation'])) for sp in spawn_points]
        blueprints = self.world.get_blueprint_library()
        bp_car = blueprints.filter(vehicle_name)[0]   
        self.veh = self.world.spawn_actor(bp_car, self.spawn_points[0])
        self.episode = 0
        self.change_ep = change_points_ep
        self.mode = mode
        self.traveled_dist = 0

        super().__init__(wrapped_world,self.veh,tag='car')

    def apply_mode(self,mode):
        if mode in ["static","random","change_point"]:
            self.mode = mode
        else:
            print(f"no mode {mode} avilable")

    def get_xy_velocity(self):
        velocity_vector = self.veh.get_velocity()
        vx = velocity_vector.x
        vy = velocity_vector.y
        velocity_magnitude_xy = math.sqrt(vx**2 + vy**2)

        return velocity_magnitude_xy


    def get_carla_actor(self):

        return self.veh
    
    @staticmethod
    def calculate_distance(prev_pos:tuple,cur_pos:tuple):
        """Calculate Euclidean distance between two positions."""

        return math.sqrt((prev_pos[0] - cur_pos[0])**2 + (prev_pos[1] - cur_pos[1])**2)
    
    def reset(self):
        """
        teleport the car 
        """
        self.episode+=1

        self.select_point()
        self.veh.set_simulate_physics(False)
        self.veh.set_transform(self.spawn_points[self.st])
        time.sleep(0.1)
        self.veh.set_simulate_physics(True)

    def select_point(self):
        """
        if it reach another points or reach change points ep it will change start point to the next point

        """
        if self.mode == "static":
            pass

        elif self.mode == "change_point":
            if self.episode % self.change_ep ==0:
                self.current_point=self.current_point+1
            self.st = self.current_point%len(self.spawn_points)

        elif self.mode == "random":
            self.st = random.choice(range(len(self.spawn_points)))



class ManageActors:

    """
    reset,destroy and give observation image all actors at once
    """
    
    def __init__(self):
        self.actor_list = []
        self.observer_list = []

    def append_observer(self,actor):
        self.observer_list.append(actor)
        self.actor_list.append(actor)

    def get_all_obs(self):

        obs = []
        
        for actor in self.observer_list:
            obs.append(actor.get_obs())

        return obs

    def append(self,actor):

        self.actor_list.append(actor)

    def reset_actors(self):

        for actor in self.actor_list:
            actor.reset()

    def destroy_actors(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()


class World(ManageActors):

    def __init__(self,
                 host,
                 port,
                 delta_frame,
                 default_weather=carla.WeatherParameters.ClearNoon,
                 sync_mode=True):
        
        self.delta_frame = delta_frame
        self.client = carla.Client(host, port)
        self.client.set_timeout(120)
        self.world = self.client.get_world()  
        self.map = self.world.get_map()
        # setting world ============================================================================
        if sync_mode:
            self.set_synchronous()

        self.world.set_weather(default_weather)
        
        super().__init__()

    def set_synchronous(self):
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.delta_frame
        settings.synchronous_mode = True
        settings.max_substeps = 16
        settings.max_substep_delta_time = 0.0125
        self.world.apply_settings(settings)
        self.client.reload_world(False)

    def set_asynchornous(self):
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

    def reset(self):
        self.destroy_actors()
        self.set_asynchornous()

    def random_wather(self):

        # carla.WeatherParameters.ClearNoon,
        # carla.WeatherParameters.ClearSunset,
        # carla.WeatherParameters.CloudyNoon,
        # carla.WeatherParameters.CloudySunset,
        # carla.WeatherParameters.WetNoon,
        # carla.WeatherParameters.WetSunset,
        # carla.WeatherParameters.MidRainyNoon,
        # carla.WeatherParameters.MidRainSunset,
        # carla.WeatherParameters.WetCloudyNoon,
        # carla.WeatherParameters.WetCloudySunset,
        # carla.WeatherParameters.HardRainNoon,
        # carla.WeatherParameters.HardRainSunset,
        # carla.WeatherParameters.SoftRainNoon,
        # carla.WeatherParameters.SoftRainSunset

        weather_presets = [
            carla.WeatherParameters.ClearNoon,
            carla.WeatherParameters.ClearSunset,
            carla.WeatherParameters.CloudyNoon,
            carla.WeatherParameters.CloudySunset,
            carla.WeatherParameters.WetNoon,
            carla.WeatherParameters.WetSunset,
        ]

        chosen_weather = random.choice(weather_presets)
        self.world.set_weather(chosen_weather)

    def get_carla_world(self):
        return self.world
    

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)