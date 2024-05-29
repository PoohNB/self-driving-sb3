import pygame
import carla
from pygame.locals import K_ESCAPE,K_TAB

class PygameControllor:

    def __init__(self,spectator_config,env):
        self.env = env
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((spectator_config['attribute']['image_size_x'], spectator_config['attribute']['image_size_y']), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

    def receive_key(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.env.close()
                    self.env.manual_end = True
                elif event.key == pygame.K_TAB:
                    self.env.spectator.change_perception()

    def render(self,image):
        # Tick render clock
        self.clock.tick()
        # self.hud.tick(self.world, self.clock)
        self.display.blit(pygame.surfarray.make_surface(image.swapaxes(0, 1)), (0, 0))

        pygame.display.flip()

    def close(self):
        pygame.close()
    