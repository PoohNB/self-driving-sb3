import cv2
import numpy as np
from scipy import ndimage
import carla
import random
import matplotlib.pyplot as plt
import os
import json
from typing import List, Tuple, Dict,Union
from environment.tools.actor_wrapper import VehicleActor,PedestrianActor

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

# class LocateObject:

#     def __init__(self, 
#                  map_path, 
#                  scale=8.596200822454035, 
#                  ref_point=(4005, 6864)):
#         self.m = scale  # Scaling factor for converting coordinates meter to pixel
#         self.ref_point = ref_point  # Reference point on the map image

#         # Load the map image and get its dimensions
#         self.map_img = cv2.imread(map_path)
#         self.map_shape = self.map_img.shape[:2]

#         self.shapes_file = os.path.join(os.path.dirname(__file__),"save/obj_sizes.json")
#         self.load_vehicle_shapes()
#         self.object_list = []
#         self.click_history = []

#     def add_object(self, obj_sizes):
#         # Add the object size data to self.vehicle_shape
#         # obj_sizes is dict that contains size of each obj
#         # example: {"obj1": (width, length), ...}
#         self.vehicle_shapes.update(obj_sizes)
#         self.save_vehicle_shapes()

#     def place_on_map(self, obj_info):
#         # obj_loc is a tuple that contains name and location
#         # example: (name, (x, y), yaw, color,call_areas)
#         if obj_info[0] not in self.vehicle_shapes.keys():
#             raise Exception(f"No object named {obj_info[0]}")
#         if len(obj_info) ==4:
#             obj_info.append(None)
#         if len(obj_info) != 5 or not isinstance(obj_info[0],str) or len(obj_info[1]) != 2 or not isinstance(obj_info[2],int):
#             raise Exception(f"the obj_info have to be in this format (name, (x, y), yaw, color) not {obj_info}")
#         if obj_info[3] == "red":
#             obj_info[3] = (0,0,255)
#         elif obj_info[3] == "blue":
#             obj_info[3] = (255,0,0)
#         elif obj_info[3] == "green":
#             obj_info[3] = (0,255,0)
#         else:
#             if len(obj_info[3]) != 3:
#                 raise Exception("invalid color")
#         self.object_list.append(obj_info)

#     def plot(self,show_index=False,call_area=False):
#         # Create a copy of the map to plot the objects on
#         map_copy = self.map_img.copy()

#         for idx,obj in enumerate(self.object_list):
#             name, (x, y), yaw , color, call_list = obj
#             width, length = self.vehicle_shapes[name]

#             # Convert position from meters to pixels
#             x_pix = int(self.ref_point[1] + x * self.m)
#             y_pix = int(self.ref_point[0] + y * self.m)  # Assuming y increases downwards in the image

#             # Convert size from meters to pixels
#             width_pix = int(width * self.m)
#             length_pix = int(length * self.m)

#             # Calculate the rectangle corners
#             rect_corners = cv2.boxPoints(((x_pix, y_pix), (length_pix, width_pix), yaw))
#             rect_corners = np.int0(rect_corners)

#             # Draw the rectangle on the map
#             cv2.drawContours(map_copy, [rect_corners], 0, color, -1)  # Red color for objects

#             if call_area:
#                 if call_list is not None: 
#                     for rad in call_list:
#                         cv2.circle(map_copy, (x_pix, y_pix), int(rad*self.m), (255,255,255),1)

#             # Label the object with its index
#             if show_index:
#                 label_position = (x_pix + 7, y_pix - 7)  # Adjust label position
#                 cv2.putText(map_copy, str(idx + 1), label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)


#         # Convert BGR image to RGB
#         self.latest_plot = cv2.cvtColor(map_copy, cv2.COLOR_BGR2RGB)

#         # Display the image
#         plt.figure(figsize=(15, 15))
#         plt.imshow(self.latest_plot)
#         plt.title('Map with Objects')
#         plt.axis('off')
#         plt.show()

#     def get_click_history(self):
#         return [(round(x, 2), round(y, 2)) for x, y in self.click_history]
    
#     def show_map_with_click(self):
#         def update_view(x, y):
#             """ Update the displayed map region based on the trackbar positions. """
#             map_copy = self.map_img[y:y+viewport_height, x:x+viewport_width]
#             cv2.imshow('Map', map_copy)

#         def mouse_callback(event, x, y, flags, param):
#             if event == cv2.EVENT_LBUTTONDOWN:
#                 x_global = x + cv2.getTrackbarPos('Horizontal', 'Map')
#                 y_global = y + cv2.getTrackbarPos('Vertical', 'Map')
#                 x_meters = (x_global - self.ref_point[1]) / self.m
#                 y_meters = (y_global - self.ref_point[0]) / self.m
#                 print(f"Clicked at pixel: ({x_global}, {y_global}), meters: ({x_meters:.2f}, {y_meters:.2f})")
#                 self.click_history.append((x_meters,y_meters))
                
#                 # Draw a small red circle at the clicked location
#                 cv2.circle(self.map_img, (x_global, y_global), 5, (0, 0, 255), -1)
#                 update_view(cv2.getTrackbarPos('Horizontal', 'Map'), cv2.getTrackbarPos('Vertical', 'Map'))

#         max_x = self.map_img.shape[1]
#         max_y = self.map_img.shape[0]
#         viewport_width = 800  # Width of the visible region
#         viewport_height = 600  # Height of the visible region

#         cv2.namedWindow('Map')
#         cv2.createTrackbar('Horizontal', 'Map', 0, max_x - viewport_width, lambda x: update_view(x, cv2.getTrackbarPos('Vertical', 'Map')))
#         cv2.createTrackbar('Vertical', 'Map', 0, max_y - viewport_height, lambda y: update_view(cv2.getTrackbarPos('Horizontal', 'Map'), y))
        
#         cv2.setMouseCallback('Map', mouse_callback)
        
#         # Initial view update
#         update_view(0, 0)

#         # Wait until a key is pressed to close the window
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


#     def clear_object(self):
#         self.object_list = []

#     def remove_veh(self,name):
#         self.vehicle_shapes.remove(name)
#         self.save_vehicle_shapes()

#     def clear_all_veh(self):
#         self.vehicle_shapes = {}
#         self.save_vehicle_shapes()

#     def get_latest_plot(self):
#         return self.latest_plot
    
#     def save_vehicle_shapes(self):
#         with open(self.shapes_file, 'w') as f:
#             json.dump(self.vehicle_shapes, f, indent=4)

#     def load_vehicle_shapes(self):
#         if os.path.exists(self.shapes_file):
#             with open(self.shapes_file, 'r') as f:
#                 self.vehicle_shapes = json.load(f)
#         else:
#             self.vehicle_shapes = {'box': (2, 2)}
#             self.save_vehicle_shapes()



class LocateObject:
    def __init__(self, map_path: str, scale: float = 8.596200822454035, ref_point: tuple = (4005, 6864)):
        self.m = scale
        self.ref_point = ref_point
        self.map_img = cv2.imread(map_path)
        if self.map_img is None:
            raise FileNotFoundError(f"Map image not found at path: {map_path}")
        self.map_shape = self.map_img.shape[:2]
        self.map_copy = self.map_img.copy()  # Create a copy of the original map for drawing
        self.shapes_file = os.path.join(os.path.dirname(__file__), "save/obj_sizes.json")
        self.load_vehicle_shapes()
        self.object_list = []
        self.click_history = []
        self.current_color = (0, 0, 255)  # Default color is red
        self.dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),(255, 255, 0),(0, 255, 255),(255, 0, 255)]  # Red, Green, Blue

    def add_object(self, obj_sizes: dict):
        self.vehicle_shapes.update(obj_sizes)
        self.save_vehicle_shapes()

    def place_on_map(self, obj_info: tuple):
        if obj_info[0] not in self.vehicle_shapes.keys():
            raise ValueError(f"No object named {obj_info[0]}")
        if len(obj_info) == 4:
            obj_info = (*obj_info, None)
        if len(obj_info) != 5 or not isinstance(obj_info[0], str) or len(obj_info[1]) != 2 or not isinstance(obj_info[2], int):
            raise ValueError(f"Object info must be in the format (name, (x, y), yaw, color) not {obj_info}")
        if isinstance(obj_info[3], str):
            color_mapping = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}
            obj_info = (obj_info[0], obj_info[1], obj_info[2], color_mapping.get(obj_info[3], (0, 0, 0)), obj_info[4])
        elif len(obj_info[3]) != 3:
            raise ValueError("Invalid color")
        self.object_list.append(obj_info)

    def plot(self, show_index: bool = False, call_area: bool = False):
        map_copy = self.map_img.copy()
        for idx, obj in enumerate(self.object_list):
            name, (x, y), yaw, color, call_list = obj
            width, length = self.vehicle_shapes[name]
            x_pix = int(self.ref_point[1] + x * self.m)
            y_pix = int(self.ref_point[0] + y * self.m)
            width_pix = int(width * self.m)
            length_pix = int(length * self.m)
            rect_corners = cv2.boxPoints(((x_pix, y_pix), (length_pix, width_pix), yaw))
            rect_corners = np.int0(rect_corners)
            cv2.drawContours(map_copy, [rect_corners], 0, color, -1)
            if call_area and call_list:
                for rad in call_list:
                    cv2.circle(map_copy, (x_pix, y_pix), int(rad * self.m), (255, 255, 255), 1)
            if show_index:
                label_position = (x_pix + 7, y_pix - 7)
                cv2.putText(map_copy, str(idx + 1), label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        self.latest_plot = cv2.cvtColor(map_copy, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 15))
        plt.imshow(self.latest_plot)
        plt.title('Map with Objects')
        plt.axis('off')
        plt.show()

    def get_click_history(self) -> dict:
        """
        Get the click history organized by color.

        :return: Dictionary with colors as keys and lists of positions as values.
        """
        color_to_positions = {color: [] for color in self.dot_colors}
        for x, y, color in self.click_history:
            color_to_positions[color].append((round(x, 2), round(y, 2)))
        return color_to_positions

    def show_map_with_click(self):
        def update_view(x, y):
            map_copy = self.map_copy[y:y + viewport_height, x:x + viewport_width]
            cv2.imshow('Map', map_copy)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                x_global = x + cv2.getTrackbarPos('Horizontal', 'Map')
                y_global = y + cv2.getTrackbarPos('Vertical', 'Map')
                x_meters = (x_global - self.ref_point[1]) / self.m
                y_meters = (y_global - self.ref_point[0]) / self.m
                print(f"Clicked at pixel: ({x_global}, {y_global}), meters: ({x_meters:.2f}, {y_meters:.2f})")
                self.click_history.append((x_meters, y_meters, self.current_color))  # Append with current color

                cv2.circle(self.map_copy, (x_global, y_global), 5, self.current_color, -1)
                update_view(cv2.getTrackbarPos('Horizontal', 'Map'), cv2.getTrackbarPos('Vertical', 'Map'))

            elif event == cv2.EVENT_MBUTTONDOWN:
                x_global = x + cv2.getTrackbarPos('Horizontal', 'Map')
                y_global = y + cv2.getTrackbarPos('Vertical', 'Map')
                for idx, (x_m, y_m, color) in enumerate(self.click_history):
                    x_pix = int(self.ref_point[1] + x_m * self.m)
                    y_pix = int(self.ref_point[0] + y_m * self.m)
                    if (x_global - x_pix) ** 2 + (y_global - y_pix) ** 2 <= 25:  # Check if the click is within the dot
                        # Restore the original map area where the dot was
                        self.map_copy[y_pix-6:y_pix+6, x_pix-6:x_pix+6] = self.map_img[y_pix-6:y_pix+6, x_pix-6:x_pix+6]
                        del self.click_history[idx]
                        update_view(cv2.getTrackbarPos('Horizontal', 'Map'), cv2.getTrackbarPos('Vertical', 'Map'))
                        break

        def color_callback(pos):
            self.current_color = self.dot_colors[pos]

        max_x, max_y = self.map_img.shape[1], self.map_img.shape[0]
        viewport_width, viewport_height = 800, 600

        cv2.namedWindow('Map')
        cv2.createTrackbar('Horizontal', 'Map', 0, max_x - viewport_width, lambda x: update_view(x, cv2.getTrackbarPos('Vertical', 'Map')))
        cv2.createTrackbar('Vertical', 'Map', 0, max_y - viewport_height, lambda y: update_view(cv2.getTrackbarPos('Horizontal', 'Map'), y))
        cv2.createTrackbar('Color', 'Map', 0, len(self.dot_colors) - 1, color_callback)

        cv2.setMouseCallback('Map', mouse_callback)

        update_view(0, 0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def clear_object(self):
        self.object_list = []

    def remove_veh(self, name: str):
        self.vehicle_shapes.pop(name, None)
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

    def __init__(self,
                 world,
                 scene_config_list: List[Dict],
                 rest_area: Union[List,Tuple]):

        self.world = world
        self.carla_map = self.world.get_map()

        self.configs = scene_config_list

        self.actor_names = self.get_names()
        
        self.road_area_lookups = []
        self.scenes = []
        c = 0
        self.max_values = 0
        for cf in self.configs:
            self.road_area_lookups.extend([self.roadpoint_preprocess(loc) for loc in cf['available_loc']])
            idx_list = list(range(c, c+len(cf['available_loc'])))
            c += len(cf['available_loc'])
            if cf['values'] > self.max_values:
                self.max_values = cf['values']
            self.scenes.append({'idx': idx_list, 'values': cf['values'], 'on_road_ratio': cf['on_road_ratio']})
        self.rest_area_lookups = self.restpoint_preprocess(rest_area)
        if self.max_values > len(self.rest_area_lookups):
            raise ValueError("number of actor more than rest area")
        self.occupied_rest= [-1] * len(self.rest_area_lookups)
        self.occupied_road= [-1] * len(self.road_area_lookups)
        self.spawn_actor()
    
    def spawn_actor(self):
        # self.actors = [random.choice(self.world.try_spawn(bp,)) for bp in self.actor_names]
        raise NotImplementedError("have to implement method spawn_actor in subclass")
    
    def get_names(self):
        # return self.bp.filter('vehicle.*.*')
        raise NotImplementedError("have to implement method get_names in subclass")
    
    def restpoint_preprocess(self, rest_area):
        # return rest_area
        raise NotImplementedError("have to implement method restpoint_preprocess in subclass")
    
    def roadpoint_preprocess(self,loc):
        # return loc
        raise NotImplementedError("have to implement method roadpoint_preprocess in subclass")
 

    def randomly_place(self, sample_idx: List[int]):
        """Randomly place cars or pedestrians in specified indices."""

        empty, parked = [], []
        for i, j in enumerate(self.occupied_rest):
            (empty if j < 0 else parked).append(i)

        placing_idx = []
        removing_idx = []
        for i, j in enumerate(self.occupied_road):
            if j < 0:
                if i in sample_idx:
                    placing_idx.append(i)
            else:
                if i not in sample_idx:
                    removing_idx.append(i)

        for i in placing_idx:
            if removing_idx:
                j = random.choice(removing_idx)
                removing_idx.remove(j)
                obj_idx = self.occupied_road[j]
                self.occupied_road[j] = -1
            elif parked:
                j = random.choice(parked)
                parked.remove(j)
                obj_idx = self.occupied_rest[j]
                self.occupied_rest[j] = -1
                empty.append(j)

            self.actors[obj_idx].move(self.road_area_lookups[i])
            self.occupied_road[i] = obj_idx

        for i in removing_idx:
            j = random.choice(empty)
            empty.remove(j)
            obj_idx = self.occupied_road[i]
            self.occupied_road[i] = -1
            self.actors[obj_idx].move(self.rest_area_lookups[j])
            self.occupied_rest[j] = obj_idx

    def reset(self,scene_idx:int):
        scene = self.scenes[scene_idx]
        if scene['on_road_ratio'] == -1:
            on_road_ratio = random.random()  # Generate a random ratio between 0 and 1
        else:
            on_road_ratio = scene['on_road_ratio']
        sample_idx = random.sample(scene['idx'],round(scene['values']*on_road_ratio))
        self.randomly_place(sample_idx)


class VehiclePlacer(ObjectPlacer):

    def __init__(self,
                world,
                scene_config_list: List[Dict],
                rest_area: List[Dict]):
        self.reverse = True
        super().__init__(world,scene_config_list,rest_area)
        

    def get_names(self) -> List:
        """Retrieve vehicle blueprints."""
        veh_list = [
            "vehicle.audi.etron", "vehicle.audi.tt", "vehicle.mercedes.coupe",
            "vehicle.mercedes.coupe_2020", "vehicle.mini.cooper_s_2021",
            "vehicle.tesla.model3", "vehicle.bh.crossbike",
            "vehicle.diamondback.century", "vehicle.gazelle.omafiets"
        ]
        return veh_list

    def restpoint_preprocess(self, area:List) -> carla.Transform:
        """Convert a configuration dictionary into a CARLA transform."""
        return [carla.Transform(carla.Location(*config['Location']), carla.Rotation(*config['Rotation'])) for config in area]
        

    def roadpoint_preprocess(self, loc: Tuple[float, float]) -> carla.Waypoint:
        """Get the closest waypoint to a given location."""
        carla_location = carla.Location(*loc, 0.1)
        transform = self.carla_map.get_waypoint(carla_location, project_to_road=True, lane_type=(carla.LaneType.Driving)).transform
        if self.reverse:
            transform.rotation.yaw += 180
            if transform.rotation.yaw >= 360:
                transform.rotation.yaw -= 360
        return transform

    def spawn_actor(self):
        """Randomly Select and Spawn vehicles in the environment."""
        selected_veh = [random.choice(self.actor_names) for _ in range(self.max_values)]
        self.actors = []
        for i, name in enumerate(selected_veh):
            veh = VehicleActor(self.world, name, self.rest_area_lookups[i])
            if veh is not None:
                self.actors.append(veh)
                self.occupied_rest[i] = i
    



class PedestriansPlacer(ObjectPlacer):

    def __init__(self,
            world,
            scene_config_list: List[Dict],
            rest_area: Tuple[Tuple[float, float], Tuple[float, float]]):
        
        super().__init__(world,scene_config_list,rest_area)

    
    def get_names(self):
        bp = self.world.get_blueprint_library()
        ped_bp = bp.filter('walker.pedestrian.*')

        return [blueprint.id for blueprint in ped_bp]

    def spawn_actor(self):
        """Spawn pedestrians in the environment."""
        selected_ped = [random.choice(self.actor_names) for _ in range(self.max_values)]
        self.actors = []
        for i, name in enumerate(selected_ped):
            self.actors.append(PedestrianActor(self.world, name, self.rest_area_lookups[i]))
            self.occupied_rest[i] = i

    def restpoint_preprocess(self, area: Tuple[Tuple[float, float], Tuple[float, float]], cell_size: int = 3) -> List[Tuple[float, float]]:
        """Generate a grid of available positions within the defined area."""
        top_left, bottom_right = area
        x_cells = int((bottom_right[0] - top_left[0]) // cell_size)
        y_cells = int((bottom_right[1] - top_left[1]) // cell_size)
        transforms = []
        count = 0
        for i in range(x_cells):
            for j in range(y_cells):
                transforms.append(carla.Transform(carla.Location(*(top_left[0] + i * cell_size, top_left[1] + j * cell_size, 0.2))))
                count += 1
                if count >= self.max_values:
                    break
            if count >= self.max_values:
                break
        return transforms
    
    def roadpoint_preprocess(self, loc: Tuple[float, float]) -> carla.Transform:
        """Generate a random transform for a pedestrian at a given position."""
        return carla.Transform(
            carla.Location(*loc, 0.1),
            carla.Rotation(0, random.randint(0, 359), 0)
        )
        
        

