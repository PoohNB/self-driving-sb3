 

def mc_command(self, **kwargs):
    """
    Convert action values to motor controller command.

    Args:
        **kwargs: Arbitrary keyword arguments, expected to contain 'action'.

    Returns:
        list: Motor controller command.
    """
    steer, throttle, brake = self.__call__(**kwargs)

    steer = int(((steer + 1) / 2) * 255)
    throttle = int(throttle * 255)
    brake = 255 if brake > 127 else 0

    if steer > 0 and throttle > 0:
        throttle = 0
        print("Brake and throttle applied at the same time!") 

    return [36, steer, throttle, brake, 0, 64]


class OriginAction:
    def __init__(self, steer_still_range = 0):
        self.steer_still_range=steer_still_range
        self.previous_steer=0

    def __call__(self, action):

        if abs(action[0] - self.previous_steer) < self.steer_still_range:
            action[0] = self.previous_steer

        self.previous_steer = action[0]
        return float(action[0]), float(action[1]), False


class LimitAction:
    def __init__(self, 
                 throttle_range :tuple,
                 max_steer = 0.8,
                 steer_still_range = 0):
        self.throttle_range =throttle_range
        self.max_steer = max_steer
        self.steer_still_range = steer_still_range
        self.previous_steer=0

    def __call__(self, action):

        steer, throttle = action

        new_steer = steer * self.max_steer
        new_throttle = throttle * (self.throttle_range[1] - self.throttle_range[0]) + self.throttle_range[0]

        if abs(new_steer - self.previous_steer) < self.steer_still_range:
                new_steer = self.previous_steer

        self.previous_steer = new_steer
        return float(new_steer), float(new_throttle), 0

    
class DiscretizeAction:

    def __init__(self, steer_list,throttle_list,steer_still_range):
        self.steer_list = steer_list
        self.throttle_list = throttle_list
        self.steer_still_range = steer_still_range
        self.previous_steer=0

    def __call__(self,action):
        # this convert continuous action to discret
        # self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

        steer_list = self.steer_list
        throttle_list = self.throttle_list
        steer,throttle = action

        steer_range = (2/len(steer_list))-1
        throttle_range = 1/len(throttle_list)

        for i in range(len(steer_list)):
            if steer < (i+1)*steer_range:
                new_steer = steer_list[i]

        for i in range(len(throttle_list)):
            if throttle < (i+1)*throttle_range:
                new_throttle = throttle_list[i]
        
        if abs(new_steer - self.previous_steer) < self.steer_still_range:
            new_steer = self.previous_steer
        
        self.previous_steer = new_steer
        return float(new_steer),float(new_throttle),0

