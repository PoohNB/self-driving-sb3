
spawn = [(-166.7, -14.31), (-143.44, -19.43), (-121.8, -24.08)]

call_rad = (15,40)
cmd_points = dict( name = "DirectionCmd",
                        configs=[dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "left"),
                                dict(loc=(0.0, -1.51),
                                   call_rad = call_rad,
                                   cmd = "left")]
                  )



placer_car = dict(available_loc = [(-18.85, -4.65),
                                (-45.49, -19.54),
                                (-80.27, -28.15),
                                (-122.73, -20.36),
                                (-159.02, -12.68),
                                (-185.55, -7.21)],
                         values = 6,
                         on_road_ratio = -1
                         )
                         

placer_ped = dict(available_loc = [],
                         on_road_ratio = 0,
                        values=5)


from config.env.reward_config import aic2admin
reward_config = aic2admin
