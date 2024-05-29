 
class ActionWrapper:

    def __init__(self,action_config):

        self.action_config = action_config

    def __call__(self,**arg):
        
        # return steer,throttle,brake

        raise NotImplementedError("have to imprement __call__ in subclass")
    
    def mc_command(self,**arg):

        """
        Convert action values to motor controller command.
        
        Args:
            **arg: Arbitrary keyword arguments, expected to contain 'action'.
        
        Returns:
            list: Motor controller command.
        """
        
        # steer -> [-1,1], throttle -> [0,1], brake -> [0,1]
        steer,throttle,brake =self(arg)

        steer = int(((steer+1)/2)*255)
        throttle = int(throttle*255)
        brake = int(brake*255)

        if steer >0 and throttle >0:
            throttle = 0
            print("brake and throttle at the same time!") 

        return [36,steer,throttle,brake,0,64]
    
class OriginAction(ActionWrapper):

    def __init__(self):

        pass

    def __call__(self,**arg):

        action = arg["action"]

        return float(action[0]),float(action[1]),0

class LimitAction(ActionWrapper):

    def __init__(self,action_config):

        self.action_config = action_config

    def __call__(self,**arg):
        # this limit the range of action
        # self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

        action = arg["action"]
        steer_limit = self.action_config['max_steer']
        throttle_limit = self.action_config['throttle_range']
        # steer_limit = arg["steer_limit"] 
        # throttle_limit = arg["throttle_limit"] 

        steer,throttle = action

        new_steer = steer*steer_limit
        new_throttle = throttle*(throttle_limit[1]-throttle_limit[0])+throttle_limit[0]
        
        return float(new_steer),float(new_throttle),0
    
class DiscretizeAction(ActionWrapper):

    def __init__(self,action_config):

        self.action_config = action_config

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
        
        return float(new_steer),float(new_throttle),0

