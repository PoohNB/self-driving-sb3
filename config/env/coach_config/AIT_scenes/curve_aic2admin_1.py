
spawn = [(-350.04, -130.52), (-352.6, -110.05), (-352.95, -73.64)]

call_rad = (15,40)
cmd_points = dict( name = "DirectionCmd",
                        configs=[dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "left"),
                                dict(loc=(0.0, -1.51),
                                   call_rad = call_rad,
                                   cmd = "left")]
                  )



placer_car = dict(available_loc = [(-355.74, -118.77),
                                    (-358.18, -87.02),
                                    (-353.64, -50.6),
                                    (-334.22, -23.5),
                                    (-346.55, -35.95),
                                    (-315.26, -7.56),
                                    (-285.59, 4.54),
                                    (-253.25, 5.0),
                                    (-216.96, -1.63)],
                         values = 8,
                         on_road_ratio = -1
                         )
                         

placer_ped = dict(available_loc = [],
                         on_road_ratio = 0,
                        values=5)


from config.env.reward_config import aic2admin
reward_config = aic2admin