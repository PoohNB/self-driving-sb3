action_limit = dict(name="LimitAction",
                    config=dict(throttle_range = (0.0,0.32),
                                max_steer = 0.6,
                                activate_filter_8bit=True))

action_original = dict(name="ActionBase",
                    config=dict())

discrete_actions1 = [[-0.6,0.4],[-0.05,0.56],[0,0.6],[0,0.4],[0,0],[0.05,0.56],[0.6,0.4]]