# sensor.camera.semantic_segmentation
# sensor.camera.rgb
# resolution 720x1280
import carla

Attachment = carla.AttachmentType

camera = dict(type = "sensor.camera.rgb",
              attribute= dict(
                 image_size_x=1280,
                 image_size_y=720,
                 fov=78,
                 sensor_tick=0.001),
               AttachmentType = Attachment.Rigid,
               tag="obs")

front_cam = dict(name='front_camera',
                 **camera,
                 Location=(0.98, 0, 1.675),
                 Rotation=(-10, 0, 0)#12.5
                 )

left_cam =  dict(name='left_camera',
                 **camera,                 
                 Location=(0, -0.61, 1.675),
                 Rotation=(-30, -90, 0)
                 )

right_cam =  dict(name='right_camera',
                 **camera,                 
                 Location=(0, 0.61, 1.675),
                 Rotation=(-30, 90, 0)
                 )

back_cam = dict(name='back_camera',
                 **camera,               
                 Location=(-0.98, 0, 1.675),
                 Rotation=(-12.5, 180, 0)
                 )

front_cam_seg = dict(front_cam,type = 'sensor.camera.semantic_segmentation')

left_cam_seg = dict(left_cam,type = 'sensor.camera.semantic_segmentation')

right_cam_seg = dict(right_cam,type = 'sensor.camera.semantic_segmentation')

back_cam_seg = dict(back_cam,type = 'sensor.camera.semantic_segmentation')

spectator_cam = dict(type = "sensor.camera.rgb",
                attribute= dict(
                 image_size_x=1280,
                 image_size_y=720))





