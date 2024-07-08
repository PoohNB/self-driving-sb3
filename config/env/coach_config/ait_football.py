# spawn_point============================================================================
# spawn_outer = [
#              dict(Location=(-573.5,-223.5,0.2),Rotation=(0,270,0)),
#              dict(Location=(-490.1,-369.2,0.2),Rotation=(0,0,0)),
#              dict(Location=(-347.4,-273.5,0.2),Rotation=(0,90,0)),
#              dict(Location=(-361,-142.9,0.2),Rotation=(0,180,0)),
#              dict(Location=(-561.1,-149.3,0.2),Rotation=(0,182,0))
             
#              ]

# spawn_inner= [
#              dict(Location=(-480.0,-366.0,0.2),Rotation=(0,180,0)),
#              dict(Location=(-570.0,-280.0,0.2),Rotation=(0,90,0)),
#              dict(Location=(-518.0,-151.0,0.2),Rotation=(0,0,0)),
#              dict(Location=(-351.4,-271.0,0.2),Rotation=(0,270,0)),             
#              ]

spawn_outer_long = [
                  (-573.5,-223.5),
                  (-490.1,-369.2),
                  (-347.4,-273.5),
                  (-361,-142.9),
                  (-561.1,-149.3),
                  
                  ]

spawn_inner_long= [
                  (-480.0,-366.0),
                  (-570.0,-280.0),
                  (-518.0,-151.0),
                  (-351.4,-271.0),             
                  ]


spawn_outer_short= [
                   (-572.46, -345.85),
                   (-509.88, -369.35),
                   (-374.35, -367.26),
                   (-346.66, -329.45),
                   (-348.29, -174.73),
                   (-374.70, -142.97),
                   (-536.05, -148.90),
                   ]

spawn_inner_short= [(-530.12, -366.21),
                  (-451.71, -365.86),
                  (-350.04, -342.24),
                  (-351.9, -236.73),
                  (-383.54, -145.88),
                  (-570.37, -176.94),
                  (-568.62, -342.71)]

# command point ==========================================================================

junction =[(-576.6548461914062, -148.419189453125),
            (-570.05419921875, -366.15301513671875),
            (-347.0089111328125, -364.6258544921875),
            (-350.5668029785156, -143.1997528076172)]


call_rad = (15,40)
# -1,0,1 => left,forward,right
cmd_points_inner = dict( name = "DirectionCmd",
                         configs=[dict(loc=(-576.6548461914062, -148.419189453125),
                                   call_rad = call_rad,
                                   cmd = "left"),
                                dict(loc=(-570.05419921875, -366.15301513671875),
                                   call_rad = call_rad,
                                   cmd = "left"),
                                dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "left"),
                                dict(loc=(-350.5668029785156, -143.1997528076172),
                                   call_rad = call_rad,
                                   cmd = "left"),
                                   ])

cmd_points_outer = dict( name = "DirectionCmd",
                        configs=[dict(loc=(-576.6548461914062, -148.419189453125),
                                   call_rad = call_rad,
                                   cmd = "right"),
                                dict(loc=(-570.05419921875, -366.15301513671875),
                                   call_rad = call_rad,
                                   cmd = "right"),
                                dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = "right"),
                                dict(loc=(-350.5668029785156, -143.1997528076172),
                                   call_rad = call_rad,
                                   cmd = "right"),
                                   ])
command_dict = dict(forward=[0],left=[1],right=[2])
default_cmd = "forward"
cmd_guide = dict(cmd_dict=command_dict,
                  default_cmd=default_cmd)
# placer config
dummy_placer = dict(available_loc = [],
                        values =0,
                        on_road_ratio=0.8
                        )

placer_car_outer = dict(available_loc = [(-568.16, -357.95),
                                       (-558.97, -366.09),
                                       (-569.32, -243.25),
                                       (-570.48, -159.14),
                                       (-561.41, -151.93),
                                       (-599.68, -152.16),
                                       (-581.77, -142.5),
                                       (-509.76, -151.0),
                                       (-451.59, -148.55),
                                       (-362.83, -145.76),
                                       (-352.13, -159.02),
                                       (-353.76, -130.76),
                                       (-339.45, -145.41),
                                       (-352.02, -237.31),
                                       (-351.2, -285.24),
                                       (-350.39, -324.21),
                                       (-349.57, -357.48),
                                       (-359.34, -363.77),
                                       (-440.78, -365.51),
                                       (-505.8, -366.56),
                                       (-593.75, -366.56),
                                       (-331.19, -366.09),
                                       (-337.59, -265.93)],
                         values = 14,
                         on_road_ratio = 0.8
                         )
                         

placer_ped_outer = dict(available_loc = [(-531.98, -152.04),
                                          (-524.65, -151.0),
                                          (-569.2, -198.46),
                                          (-569.09, -192.29),
                                          (-568.97, -273.49),
                                          (-568.74, -348.53),
                                          (-570.83, -377.96),
                                          (-546.06, -365.63),
                                          (-471.37, -365.74),
                                          (-396.92, -364.46),
                                          (-351.55, -342.13),
                                          (-352.48, -293.27),
                                          (-352.95, -260.7),
                                          (-353.06, -256.04),
                                          (-353.3, -196.13),
                                          (-412.04, -146.46),
                                          (-399.25, -146.81),
                                          (-586.54, -153.09),
                                          (-568.86, -164.49)],
                         on_road_ratio = 1,
                        values=5)

placer_car_inner = dict(available_loc = [(-586.89, -149.48),
                                          (-578.4, -143.55),
                                          (-516.97, -148.55),
                                          (-429.61, -144.48),
                                          (-362.6, -143.09),
                                          (-349.92, -130.87),
                                          (-339.92, -142.97),
                                          (-348.53, -158.33),
                                          (-347.94, -210.91),
                                          (-347.48, -283.61),
                                          (-339.68, -270.35),
                                          (-347.94, -238.71),
                                          (-345.97, -355.74),
                                          (-346.78, -318.05),
                                          (-335.61, -369.47),
                                          (-360.51, -367.02),
                                          (-432.28, -368.77),
                                          (-499.29, -369.81),
                                          (-559.55, -369.23),
                                          (-580.61, -369.35),
                                          (-572.0, -358.41),
                                          (-573.16, -270.7),
                                          (-574.44, -159.02),
                                          (-560.6, -149.25)],
                        values =14,
                        on_road_ratio=0.8
                        )

placer_ped_inner = dict(available_loc = [(-549.2, -148.67),
                                          (-541.05, -148.55),
                                          (-476.02, -146.46),
                                          (-451.71, -145.65),
                                          (-402.04, -143.09),
                                          (-382.38, -143.55),
                                          (-346.55, -175.89),
                                          (-346.32, -190.2),
                                          (-346.55, -225.91),
                                          (-346.55, -259.77),
                                          (-344.11, -305.95),
                                          (-344.8, -342.01),
                                          (-344.92, -373.42),
                                          (-384.12, -368.53),
                                          (-476.02, -370.05),
                                          (-540.82, -370.16),
                                          (-573.39, -338.52),
                                          (-573.28, -309.09),
                                          (-573.51, -222.66),
                                          (-574.32, -217.42),
                                          (-574.67, -187.18)],
                        values=5,
                         on_road_ratio = 1
                        )


# rest point config ===============================================================================
parking_available = [dict(Location=(-526.4, -144.25,0.2), Rotation=(0,90,0)),
                     dict(Location=(-531.51, -144.25,0.2), Rotation=(0,90,0)),
                     dict(Location=(-536.4, -144.37,0.2), Rotation=(0,90,0)),
                     dict(Location=(-544.08, -144.37,0.2), Rotation=(0,90,0)),
                     dict(Location=(-548.85, -144.6,0.2), Rotation=(0,90,0)),
                     dict(Location=(-553.73, -144.37,0.2), Rotation=(0,90,0)),
                     dict(Location=(-558.5, -144.6,0.2), Rotation=(0,90,0)),
                     dict(Location=(-568.27, -144.72,0.2), Rotation=(0,90,0)),
                     dict(Location=(-578.2,-160.8,0.2),Rotation=(0,219,0)),
                     dict(Location=(-578.2,-163.6,0.2),Rotation=(0,219,0)),
                     dict(Location=(-585.5,-156.1,0.2),Rotation=(0,90,0)),
                     dict(Location=(-571.1,-385.0,0.2),Rotation=(0,90,0)),
                     dict(Location=(-448.0,-382.8,0.2),Rotation=(0,270,0)),
                     dict(Location=(-454.8,-382.8,0.2),Rotation=(0,270,0))]

people_area =  ((-500.33731049714527, -324.5619847214683),
               (-452.641821702951, -286.9872459884811))


# scene config ==============================================================================
no_obsc = dict(car_obsc = dummy_placer,
               ped_obsc = dummy_placer)

full_obsc_outer =  dict(car_obsc = placer_car_outer,
                        ped_obsc = dummy_placer)

full_obsc_inner = dict(car_obsc = placer_car_inner,
                        ped_obsc = dummy_placer)

reward_inner = dict(name="RewardPath",
                    config = dict(mask_path = "environment/rewardmask/ait_map/ait_fb_inner.png"))

reward_outer = dict(name="RewardPath",
                    config = dict(mask_path = "environment/rewardmask/ait_map/ait_fb_outer.png"))

scene_outer_easy = dict(spawn_points = spawn_outer_short,
                          cmd_config = cmd_points_outer,
                          **no_obsc,
                          rewarder_config = reward_outer)

scene_outer_med = dict(spawn_points = spawn_outer_long,
                          cmd_config = cmd_points_outer,
                          **no_obsc,
                          rewarder_config = reward_outer)

scene_outer_hard = dict(spawn_points = spawn_outer_long,
                          cmd_config = cmd_points_outer,
                           **full_obsc_outer,
                          rewarder_config = reward_outer)

scene_inner_easy = dict(spawn_points = spawn_inner_short,
                          cmd_config = cmd_points_inner,
                          **no_obsc,
                          rewarder_config = reward_inner)

scene_inner_med = dict(spawn_points = spawn_inner_long,
                          cmd_config = cmd_points_inner,
                          **no_obsc,
                          rewarder_config = reward_inner)

scene_inner_hard = dict(spawn_points = spawn_inner_long,
                          cmd_config = cmd_points_inner,
                          **full_obsc_inner,
                          rewarder_config = reward_inner)

# coach plan config =================================================================================================
coach_base = dict( parking_area=parking_available,
                    ped_area=people_area,
                    cmd_guide=cmd_guide
                    )

plan_hard_full = dict(scene_configs=[scene_inner_hard,scene_outer_hard],
                    **coach_base,
                    )

plan_med_outer = dict(scene_configs=[scene_outer_med],
                    **coach_base,
                    )

plan_med_inner = dict(scene_configs=[scene_inner_med],
                    **coach_base,
                    )

plan_easy_outer = dict(scene_configs=[scene_outer_easy],
                    **coach_base,
                    )

plan_easy_inner = dict(scene_configs=[scene_inner_easy],
                    **coach_base,
                    )
