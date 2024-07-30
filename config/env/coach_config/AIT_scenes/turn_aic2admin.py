
spawn = [(-320.84, -366.21), (-296.76, -366.09), (-271.63, -365.51)]

call_rad = (15,40)
cmd_points = dict( name = "DirectionCmd",
                        configs=[dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "left"),
                                dict(loc=(0.0, -1.51),
                                   call_rad = call_rad,
                                   cmd = "left")]
                  )

placer_car = dict(available_loc = [(-360.86, -367.37),
                                    (-327.59, -369.7),
                                    (-303.16, -369.47),
                                    (-349.81, -355.39),
                                    (-350.39, -330.15),
                                    (-351.2, -303.62)],
                         values = 6,
                         on_road_ratio = -1
                         )
                         

placer_ped = dict(available_loc = [],
                         on_road_ratio = 0,
                        values=5)


from config.env.reward_config import aic2admin
reward_config = aic2admin