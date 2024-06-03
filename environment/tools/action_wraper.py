 
class ActionWrapper:
    def __init__(self, action_config=None, stable_steer=False):
        self.action_config = action_config
        self.stable_steer = stable_steer
        if self.stable_steer:
            self.previous_steer = 0

    def __call__(self, **kwargs):
        """
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("You must implement __call__ in a subclass.")

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
        brake = int(brake * 255)

        if steer > 0 and throttle > 0:
            throttle = 0
            print("Brake and throttle applied at the same time!") 

        return [36, steer, throttle, brake, 0, 64]


class OriginAction(ActionWrapper):
    def __init__(self, action_config=None, stable_steer=False):
        super().__init__(action_config, stable_steer)

    def __call__(self, **kwargs):
        action = kwargs["action"]

        if self.stable_steer:
            if abs(action[0] - self.previous_steer) < self.action_config["steer_still_range"]:
                action[0] = self.previous_steer

        self.previous_steer = action[0]
        return float(action[0]), float(action[1]), 0


class LimitAction(ActionWrapper):
    def __init__(self, action_config, stable_steer=False):
        super().__init__(action_config, stable_steer)

    def __call__(self, **kwargs):
        action = kwargs["action"]
        steer_limit = self.action_config['max_steer']
        throttle_limit = self.action_config['throttle_range']

        steer, throttle = action

        new_steer = steer * steer_limit
        new_throttle = throttle * (throttle_limit[1] - throttle_limit[0]) + throttle_limit[0]

        if self.stable_steer:
            if abs(new_steer - self.previous_steer) < self.action_config["steer_still_range"]:
                new_steer = self.previous_steer

        self.previous_steer = new_steer
        return float(new_steer), float(new_throttle), 0

    
class DiscretizeAction(ActionWrapper):

    def __init__(self, action_config, stable_steer=False):
        super().__init__(action_config, stable_steer)

    def __call__(self,**arg):
        # this convert continuous action to discret
        # self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

        action = arg["action"]

        steer_list = self.action_config['steer_list']
        throttle_list = self.action_config['throttle_list']
        steer,throttle = action

        steer_range = (2/len(steer_list))-1
        throttle_range = 1/len(throttle_list)

        for i in range(len(steer_list)):
            if steer < (i+1)*steer_range:
                new_steer = steer_list[i]

        for i in range(len(throttle_list)):
            if throttle < (i+1)*throttle_range:
                new_throttle = throttle_list[i]
        
        if self.stable_steer:
            if abs(new_steer - self.previous_steer) < self.action_config["steer_still_range"]:
                new_steer = self.previous_steer
        
        return float(new_steer),float(new_throttle),0

