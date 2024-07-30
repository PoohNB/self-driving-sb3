

spawn = [(-258.37, 6.05), (-298.85, 1.16), (-226.61, 0.12)]

call_rad = (15,40)
cmd_points = dict( name = "DirectionCmd",
                        configs=[dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "right"),
                                   dict(loc=(0.0, -1.51),
                                   call_rad = call_rad,
                                   cmd = "right")]
                  )

endpoint = ()

placer_car = dict(available_loc = [(-240.34, -0.81),
                                (-282.92, 0.81),
                                (-317.58, -13.15),
                                (-345.15, -40.95),
                                (-354.34, -86.2),
                                (-350.15, -128.43),
                                (-368.53, -143.55),
                                (-348.18, -159.37),
                                (-335.03, -142.97)],
                         values = 8,
                         on_road_ratio = -1
                         )
                         

placer_ped = dict(available_loc = [],
                         on_road_ratio = 0,
                        values=5)


from config.env.reward_config import admin2aic
reward_config = admin2aic

