
import carla

def carla_point(pos_ro):
    location = carla.Location(*pos_ro[0])
    rotation = carla.Rotation(yaw=pos_ro[1])
    return carla.Transform(location, rotation)
