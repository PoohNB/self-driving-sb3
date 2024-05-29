import cv2
import numpy as np
from scipy import ndimage
import carla
import random

MAX_OBSTACLE_FIELD = 200

class locate_obstacle():
    def __init__(self):
        self.m = 8.596200822454035
        self.ref_point = (4005, 6864)

        map_img = cv2.imread('../../utils/map.png')
        self.map_shape = map_img.shape[:2]

        self.obstacle = {}
        self.obstacle_range = {}

    def add(self, name, size, space):
        # 100 = 1 meter
        # size = [width, height]
        # space = [front, back, right, left]
        self.obstacle[name] = size, space

        self.obstacle_range[name] = np.zeros((MAX_OBSTACLE_FIELD, MAX_OBSTACLE_FIELD, 3), dtype=np.uint8)
        start_point_range = (round((((size[1]/2)+space[0])/(MAX_OBSTACLE_FIELD/2)*self.m)+(MAX_OBSTACLE_FIELD/2)),
                             round((((size[0]/2)+space[2])/(MAX_OBSTACLE_FIELD/2)*self.m)+(MAX_OBSTACLE_FIELD/2)))
        end_point_range = (round((((-size[1]/2)-space[1])/(MAX_OBSTACLE_FIELD/2)*self.m)+(MAX_OBSTACLE_FIELD/2)),
                           round((((-size[0]/2)-space[3])/(MAX_OBSTACLE_FIELD/2)*self.m)+(MAX_OBSTACLE_FIELD/2)))
        self.obstacle_range[name] = cv2.rectangle(self.obstacle_range[name], start_point_range, end_point_range, (255, 255, 255), -1)

    def curr_position(self, name, location, yaw):
        yaw = -yaw
        point_on_img = (round(location[0]*self.m + self.ref_point[1]), round(location[1]*self.m + self.ref_point[0]))

        this_obs_range = np.copy(self.obstacle_range[name])
        this_obs_range = ndimage.rotate(this_obs_range, yaw, order=0, reshape=0, mode='nearest')
        this_obs_range = this_obs_range[:, :, 0]

        img_for_range = np.zeros(self.map_shape, dtype=np.uint8)
        img_for_range[int(point_on_img[1]-(MAX_OBSTACLE_FIELD/2)):int(point_on_img[1]+(MAX_OBSTACLE_FIELD/2)),
                      int(point_on_img[0]-(MAX_OBSTACLE_FIELD/2)):int(point_on_img[0]+(MAX_OBSTACLE_FIELD/2))] = this_obs_range

        return img_for_range
    


class ObjectPlacer:
    def __init__(self, world, object_blueprint, available_locations,seed=2024):
        """
        Initialize the ObjectPlacer.

        :param world: The CARLA world object.
        :param object_blueprint: The blueprint of the object to place.
        :param available_locations: A list of available locations where objects can be placed.
        """
        random.seed(seed)
        self.world = world
        self.object_blueprint = object_blueprint
        self.available_locations = available_locations
        self.spawned_objects = []

    def reset(self):
        """
        Reset the scene by randomly placing objects at the available locations.
        """
        # Clear previously spawned objects
        self.clear_objects()

        # Randomly choose locations to place objects
        chosen_locations = random.sample(self.available_locations, len(self.available_locations))

        # Spawn objects at the chosen locations
        for location in chosen_locations:
            transform = carla.Transform(location)
            obj = self.world.spawn_actor(self.object_blueprint, transform)
            self.spawned_objects.append(obj)

    def clear_objects(self):
        """
        Destroy all spawned objects.
        """
        for obj in self.spawned_objects:
            if obj.is_alive:
                obj.destroy()
        self.spawned_objects = []

    def set_available_locations(self, new_locations):
        """
        Update the available locations for object placement.

        :param new_locations: A new list of available locations.
        """
        self.available_locations = new_locations

