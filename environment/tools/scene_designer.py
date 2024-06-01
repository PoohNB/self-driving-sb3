import cv2
import numpy as np
from scipy import ndimage
import carla
import random
import matplotlib.pyplot as plt
import os
import json

def get_vehicle_shapes_from_blueprint(world,veh_names):

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    carlamap = world.get_map()
    vehicle_shapes = {}
    for veh in veh_names:
        # Temporarily spawn a vehicle to get its bounding box
        blueprint = vehicle_blueprints.find(veh)
        transform = carlamap.get_spawn_points()[0]
        vehicle = world.try_spawn_actor(blueprint, transform)

        if vehicle is not None:
            # Get the bounding box of the vehicle
            bounding_box = vehicle.bounding_box
            width = bounding_box.extent.y * 2
            length = bounding_box.extent.x * 2
            # height = bounding_box.extent.z * 2

            # Store the shape in the dictionary using the blueprint ID
            vehicle_name = blueprint.id
            vehicle_shapes[vehicle_name] = (width, length)

            # Destroy the temporary vehicle
            vehicle.destroy()

    return vehicle_shapes 

class LocateObject:

    def __init__(self, 
                 map_path, 
                 scale=8.596200822454035, 
                 ref_point=(4005, 6864)):
        self.m = scale  # Scaling factor for converting coordinates meter to pixel
        self.ref_point = ref_point  # Reference point on the map image

        # Load the map image and get its dimensions
        self.map_img = cv2.imread(map_path)
        self.map_shape = self.map_img.shape[:2]

        self.shapes_file = os.path.join(os.path.dirname(__file__),"save/obj_sizes.json")
        self.load_vehicle_shapes()
        self.object_list = []

    def add_object(self, obj_sizes):
        # Add the object size data to self.vehicle_shape
        # obj_sizes is dict that contains size of each obj
        # example: {"obj1": (width, length), ...}
        self.vehicle_shapes.update(obj_sizes)
        self.save_vehicle_shapes()

    def place_on_map(self, obj_loc):
        # obj_loc is a tuple that contains name and location
        # example: (name, (x, y), yaw)
        if obj_loc[0] not in self.vehicle_shapes.keys():
            raise Exception(f"No object named {obj_loc[0]}")
        self.object_list.append(obj_loc)

    def plot(self):
        # Create a copy of the map to plot the objects on
        map_copy = self.map_img.copy()

        for idx,obj in enumerate(self.object_list):
            name, (x, y), yaw = obj
            width, length = self.vehicle_shapes[name]

            # Convert position from meters to pixels
            x_pix = int(self.ref_point[1] + x * self.m)
            y_pix = int(self.ref_point[0] + y * self.m)  # Assuming y increases downwards in the image

            # Convert size from meters to pixels
            width_pix = int(width * self.m)
            length_pix = int(length * self.m)

            # Calculate the rectangle corners
            rect_corners = cv2.boxPoints(((x_pix, y_pix), (length_pix, width_pix), yaw))
            rect_corners = np.int0(rect_corners)

            # Draw the rectangle on the map
            cv2.drawContours(map_copy, [rect_corners], 0, (0, 0, 255), -1)  # Red color for objects

            # Label the object with its index
            label_position = (x_pix + 7, y_pix - 7)  # Adjust label position
            cv2.putText(map_copy, str(idx + 1), label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)


        # Convert BGR image to RGB
        self.latest_plot = cv2.cvtColor(map_copy, cv2.COLOR_BGR2RGB)

        # Display the image
        plt.figure(figsize=(15, 15))
        plt.imshow(self.latest_plot)
        plt.title('Map with Objects')
        plt.axis('off')
        plt.show()


    def clear_object(self):
        self.object_list = []

    def remove_veh(self,name):
        self.vehicle_shapes.remove(name)
        self.save_vehicle_shapes()

    def clear_all_veh(self):
        self.vehicle_shapes = {}
        self.save_vehicle_shapes()

    def get_latest_plot(self):
        return self.latest_plot
    
    def save_vehicle_shapes(self):
        with open(self.shapes_file, 'w') as f:
            json.dump(self.vehicle_shapes, f, indent=4)

    def load_vehicle_shapes(self):
        if os.path.exists(self.shapes_file):
            with open(self.shapes_file, 'r') as f:
                self.vehicle_shapes = json.load(f)
        else:
            self.vehicle_shapes = {'box': (2, 2)}
            self.save_vehicle_shapes()




class ObjectPlacer:
    def __init__(self, world, object_blueprint, available_locations, density=0.5):
        """
        Initialize the ObjectPlacer.

        :param world: The CARLA world object.
        :param object_blueprint: The blueprint of the object to place.
        :param available_locations: A list of available locations where objects can be placed.
        :param num_objects: The number of objects to spawn and manage.
        """
        if density >1:
            raise Exception("density range is 0 to 1")
        elif density == 0:
            print("no object spawn")

        self.world = world
        self.object_blueprint = object_blueprint
        self.available_locations = available_locations
        self.num_objects = int(len(self.available_locations)*density)
        self.spawned_objects = []
        self._spawn_objects()

    def _spawn_objects(self):
        """
        Spawn objects at the start and keep references to them.
        """

        random_num = random.sample(range(len(self.available_locations)), self.num_objects)
        for n in random_num:
            transform = carla.Transform(carla.Location(*self.available_locations['Location'][n]), carla.Rotation(*self.available_locations['Rotation'][n]))
            obj = self.world.spawn_actor(self.object_blueprint, transform)
            self.spawned_objects.append(obj)

    def reset(self):
        """
        Reset the scene by moving objects to new random locations.
        """
        chosen_locations = random.sample(self.available_locations, self.num_objects)
        for obj, location in zip(self.spawned_objects, chosen_locations):
            transform = carla.Transform(location)
            obj.set_transform(transform)

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


