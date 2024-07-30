spawn = [(-350.27, -345.27), (-350.62, -331.77), (-350.85, -314.67)]

call_rad = (15,40)
cmd_points = dict( name = "DirectionCmd",
                        configs=[dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "right"),
                                   dict(loc=(0.0, -1.51),
                                   call_rad = call_rad,
                                   cmd = "right")]
                  )


placer_car = dict(available_loc = [(-345.97, -355.04),
                                    (-346.2, -330.96),
                                    (-362.37, -364.0),
                                    (-332.82, -365.63),
                                    (-301.3, -365.86),
                                    (-243.01, -365.51),
                                    (-181.24, -365.16),
                                    (-113.54, -364.93)],
                         values = 6,
                         on_road_ratio = -1
                         )
                         

placer_ped = dict(available_loc = [],
                         on_road_ratio = 0,
                        values=5)


from config.env.reward_config import admin2aic
reward_config = admin2aic
