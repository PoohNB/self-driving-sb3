sr = 0.1 # have to lower than 0.1



limitaction1 = dict(name="LimitAction",
                    config=dict(action_config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.8,
                                                    steer_still_range = sr),
                                stable_steer=True))

dicretize_action1 = dict(steer_list = [-0.6,-0.1,0,0.1,0.6],
                  throttle_list = [0.4,0.8],
                  steer_still_range = sr)


steer_speed = 0.2