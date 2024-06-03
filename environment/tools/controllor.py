import pygame
import carla
from pygame.locals import K_ESCAPE,K_TAB
from environment.tools.hud import HUD

class PygameControllor:

    def __init__(self,spectator_config,env):
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

    def receive_key(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.env.close()
                    self.env.manual_end = True
                elif event.key == pygame.K_TAB:
                    self.env.spectator.change_perception()

    def render(self,image,extra_info):
        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.env.world, self.clock)
        
        self.display.blit(pygame.surfarray.make_surface(image.swapaxes(0, 1)), (0, 0))
        self.hud.render(self.display, extra_info=extra_info)
        pygame.display.flip()

    def close(self):
        pygame.quit()
    