limitaction1 = dict(name="LimitAction",
                    config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.8,
                                                    steer_still_range = 0.1))

limitaction2 = dict(name="LimitAction",
                    config=dict(throttle_range = (0.0,0.6),
                                                    max_steer = 0.6,
                                                    steer_still_range = 0))


originaction = dict(name="OriginAction",
                    config=dict(steer_still_range = 0))

discret_actions= [[-0.6,0.4],[-0.05,0.56],[0,0.6],[0,0.4],[0,0],[0.05,0.56],[0.6,0.4]]