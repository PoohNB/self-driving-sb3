import math
import carla
from typing import Tuple,List

def calculate_distance(prev_pos:tuple,cur_pos:tuple):
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((prev_pos[0] - cur_pos[0])**2 + (prev_pos[1] - cur_pos[1])**2)

def get_correspond_waypoint_transform(carla_map, loc: Tuple[float, float],reverse=True) -> carla.Waypoint:
    """Get the closest waypoint to a given location."""

    carla_location = carla.Location(*loc, 0.1)
    transform = carla_map.get_waypoint(carla_location, project_to_road=True, lane_type=(carla.LaneType.Driving)).transform
    if reverse:
        transform.rotation.yaw += 180
        if transform.rotation.yaw >= 360:
            transform.rotation.yaw -= 360
    return transform



def carla_list_transform( area:List) -> carla.Transform:
    """Convert a configuration dictionary into a CARLA transform."""
    return [carla.Transform(carla.Location(*config['Location']), carla.Rotation(*config['Rotation'])) for config in area]
