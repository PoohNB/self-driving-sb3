from gymnasium import Wrapper

class GymWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs,_ = self.env.reset()

        return obs


    def step(self, action):
        obs, reward, terminated, _, info = self.env.step(action)
        return obs, reward, terminated, info
    
    def render(self,mode = "human"):
        return self.env.render(mode=mode)