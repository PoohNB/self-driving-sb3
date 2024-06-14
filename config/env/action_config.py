# Example


limitaction1 = dict(name="LimitAction",
                    action_config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.8,
                                                    steer_still_range = 0.1))

discretize_action1 = dict(name="DiscretizeAction",steer_list = [-0.6,-0.1,0,0.1,0.6],
                  throttle_list = [0.4,0.8],
                  steer_still_range = 0.1)

discrets_action1 = [[-0.6,0.4],[-0.1,0.56],[0,0.6],[0,0.4],[0,0],[0.1,0.56],[0.6,0.4]]

steer_speed = 0.2