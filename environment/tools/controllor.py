import pygame
from pygame.locals import K_0,K_1,K_2,K_3,K_4,K_5,K_6,K_7,K_8,K_9
from pygame.locals import K_w,K_a,K_s,K_d
from pygame.locals import K_t
import cv2
import carla
from environment.tools.hud import HUD
import numpy as np
import os
from datetime import datetime

class PygameControllor:

    def __init__(self,spectator_config,env,save_path="env_photo"):
        self.env = env()
        width = spectator_config['attribute']['image_size_x']
        height = spectator_config['attribute']['image_size_y']
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.hud = HUD(width,height)
        self.hud.set_vehicle(self.env.car)
        self.env.world.on_tick(self.hud.on_world_tick)

        self.set_photo_path(save_path)
        self.activate_control = False

    def set_photo_path(self,save_path):

        self.save_path = save_path
        os.makedirs(self.save_path,exist_ok=True)
        print(f"set save photo path to {self.save_path}")

    def init_control(self,control_list=[[-0.6,0.4],[-0.1,0.56],[0,0.6],[0,0.4],[0,0],[0.1,0.56],[0.6,0.4]]):

        self.num_keys = [K_1,K_2,K_3,K_4,K_5,K_6,K_7,K_8,K_9,K_0]
        self.wasd = [K_w,K_a,K_s,K_d]
        self.steer_spd = 1
        self.acc = 1
        self.action=[0.0,0.0]
        
        assert control_list is not None,f"control is {control_list} continous not support yet"
        assert isinstance(control_list,list),"control list have to be list"
        assert len(control_list) <= len(self.num_keys)
        self.control_list = control_list
        print(f"activated control, the control list is {self.control_list}")
        self.activate_control = True

    def receive_key(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.env.close()
                elif event.key == pygame.K_TAB:
                    self.env.spectator.change_perception()
                elif event.key == pygame.K_t:
                    raws = self.env.get_raw_images()
                    spec = self.env.get_spectator_image()
                    now = datetime.now()

                    # Format datetime as string
                    filename = now.strftime("%Y%m%d_%H%M%S")

                    for i,img in enumerate(raws):
                        raw_path = os.path.join(self.save_path,f"{filename}_raw_image_{i}.png")
                        cv2.imwrite(raw_path,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
                    spec_path = os.path.join(self.save_path,f"{filename}_spectator_image_{i}.png")
                    cv2.imwrite(spec_path,cv2.cvtColor((spec),cv2.COLOR_RGB2BGR))

                if self.activate_control:

                    if self.control_list is not None:
                        if event.key in self.num_keys:
                            idx=self.num_keys.index(event.key)
                            self.action =self.control_list[idx]
                            print(f"key press {self.action}")
                    else:
                        veh_control = self.env.car.get_control()
                        steer,throttle = veh_control.steer,veh_control.throttle
                        if event.key == K_w:
                            self.action = steer,throttle+self.acc
                        if event.key == K_a:
                            self.action = steer-self.steer_spd,throttle
                        if event.key == K_s:
                            self.action = steer,throttle-self.acc   
                        if event.key == K_d:
                            self.action = steer+self.steer_spd,throttle   

    def render(self,image,extra_info):
        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.env.world, self.clock)
        
        self.display.blit(pygame.surfarray.make_surface(image.swapaxes(0, 1)), (0, 0))
        self.hud.render(self.display, extra_info=extra_info)
        pygame.display.flip()

    def get_display_array(self):
        return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])

    def close(self):
        pygame.quit()

    

