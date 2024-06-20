import math


def calculate_distance(prev_pos:tuple,cur_pos:tuple):
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((prev_pos[0] - cur_pos[0])**2 + (prev_pos[1] - cur_pos[1])**2)