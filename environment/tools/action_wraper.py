
def action_dummy(**arg):

    action = arg["action"]

    return float(action[0]),float(action[1])



def five_action(**arg):
    # self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

    action = arg["action"]

    steer_list = [-0.6,-0.1,0,0.1,0.6]
    throttle_list = [0,0.8]
    steer,throttle = action

    if steer <-0.6:
        new_steer = steer_list[0]
    elif steer <-0.2:
        new_steer = steer_list[1]
    elif steer < 0.2:
        new_steer = steer_list[2]
    elif steer < 0.6:
        new_steer = steer_list[3]
    else:
        new_steer = steer_list[4]

    if throttle< 0.5:
        new_throttle = throttle_list[0]
    else:
        new_throttle = throttle_list[1]
    
    return new_steer,new_throttle

