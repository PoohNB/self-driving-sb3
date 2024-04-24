
import carla
import weakref
import numpy as np


class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw, custom_palette=False):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        self.custom_palette = custom_palette
        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", f"110")
        # camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):

            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if self.custom_palette:
                classes = {
                    0: [0, 0, 0],  # None
                    1: [0, 0, 0],  # Buildings
                    2: [0, 0, 0],  # Fences
                    3: [0, 0, 0],  # Other
                    4: [0, 0, 0],  # Pedestrians
                    5: [0, 0, 0],  # Poles
                    6: [157, 234, 50],  # RoadLines
                    7: [50, 64, 128],  # Roads
                    8: [255, 255, 255],  # Sidewalks
                    9: [0, 0, 0],  # Vegetation
                    10: [0, 0, 0],  # Vehicles
                    11: [0, 0, 0],  # Walls
                    12: [0, 0, 0]  # TrafficSigns
                }
                segimg = np.round((array[:, :, 0])).astype(np.uint8)
                array = array.copy()
                for j in range(array.shape[0]):
                    for i in range(array.shape[1]):
                        r_id = segimg[j, i]
                        if r_id <= 12:
                            array[j, i] = classes[segimg[j, i]]
                        else:
                            array[j, i] = classes[0]

            self.on_recv_image(array)

    def destroy(self):
        super().destroy()