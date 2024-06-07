stable_steer = 0.1 # have to lower than 0.1

limit_action = dict(throttle_range = (0.0,0.6),
                  max_steer = 0.8,
                  steer_still_range = stable_steer)

dicretize_action = dict(steer_list = [-0.6,-0.1,0,0.1,0.6],
                  throttle_list = [0.4,0.8],
                  steer_still_range = stable_steer)

steer_speed = 0.2