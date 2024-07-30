

spawn = [(-23.03, -7.56), (-39.44, -17.1), (-49.79, -21.4)]

call_rad = (15,40)
cmd_points = dict( name = "DirectionCmd",
                        configs=[dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "right"),
                                   dict(loc=(0.0, -1.51),
                                   call_rad = call_rad,
                                   cmd = "right")]
                  )


placer_car = dict(available_loc = [(-32.11, -17.8),
                                    (-56.65, -28.27),
                                    (-85.15, -32.11),
                                    (-127.15, -23.5),
                                    (-158.91, -16.29),
                                    (-184.03, -11.4)],
                         values = 6,
                         on_road_ratio = -1
                         )
                         

placer_ped = dict(available_loc = [],
                         on_road_ratio = 0,
                        values=5)


from config.env.reward_config import admin2aic
reward_config = admin2aic
