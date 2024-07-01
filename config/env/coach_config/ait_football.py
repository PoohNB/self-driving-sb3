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

placer_car_outer = dict(available_loc = [(-596.89, -367.14),
                                          (-568.51, -358.18),
                                          (-558.15, -366.32),
                                          (-493.36, -366.67),
                                          (-418.56, -365.39),
                                          (-358.53, -363.77),
                                          (-349.69, -356.44),
                                          (-318.4, -365.98),
                                          (-350.62, -317.81),
                                          (-351.55, -286.41),
                                          (-351.55, -252.9),
                                          (-352.02, -158.79),
                                          (-339.92, -143.09),
                                          (-339.92, -143.09),
                                          (-375.4, -146.11),
                                          (-355.27, -120.05),
                                          (-428.91, -147.27),
                                          (-496.73, -150.53),
                                          (-560.6, -151.69),
                                          (-570.37, -158.67),
                                          (-609.11, -152.74),
                                          (-581.77, -143.32),
                                          (-569.32, -232.66),
                                          (-568.97, -305.95)],
                         values = 4,
                         on_road_ratio = 1
                         )
                         

placer_ped_outer = dict(available_loc = [(-569.32, -172.87),
                                        (-479.05, -150.53),
                                        (-383.77, -145.76),
                                        (-575.84, -282.22),
                                        (-499.17, -145.65),
                                        (-331.77, -365.98)],
                        values=2)

placer_car_inner = dict(available_loc = [(-572.23, -358.18),
                                       (-582.23, -369.58),
                                       (-558.15, -369.58),
                                       (-573.04, -283.85),
                                       (-574.21, -159.84),
                                       (-567.81, -149.14),
                                       (-578.86, -142.74),
                                       (-588.86, -149.83),
                                       (-473.46, -146.46),
                                       (-364.93, -142.97),
                                       (-348.29, -157.74),
                                       (-350.04, -129.59),
                                       (-348.41, -249.18),
                                       (-347.01, -292.57),
                                       (-346.43, -328.52),
                                       (-345.85, -356.67),
                                       (-332.94, -365.74),
                                       (-362.49, -367.37),
                                       (-504.87, -369.47),
                                       (-446.83, -368.77)],
                        values =4,
                        on_road_ratio=0.8
                        )

placer_ped_inner = dict(available_loc = [],
                        values=0)


# rest point config ===============================================================================
parking_available = [dict(Location=(-526.3,-145.0,0.2),Rotation=(0,90,0)),
                     # dict(Location=(-528.9,-145.0,0.1),Rotation=(0,90,0)),
                     dict(Location=(-531.4,-145.0,0.2),Rotation=(0,90,0)),
                     # dict(Location=(-533.8,-145.0,0.1),Rotation=(0,90,0)),
                     dict(Location=(-540.2,-145.0,0.2),Rotation=(0,90,0)),
                     dict(Location=(-556.0,-145.0,0.2),Rotation=(0,90,0)),
                     # dict(Location=(-558.4,-145.0,0.1),Rotation=(0,90,0)),
                     dict(Location=(-570.7,-145.0,0.2),Rotation=(0,90,0)),
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
                        ped_obsc = placer_ped_outer)

full_obsc_inner = dict(car_obsc = placer_car_inner,
                        ped_obsc = placer_ped_inner)

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
rest_config = dict( parking_area=parking_available,
                    ped_area=people_area,
                    cmd_guide=cmd_guide
                    )

plan_hard_full = dict(scene_configs=[scene_inner_hard,scene_outer_hard],
                    **rest_config,
                    )

plan_med_outer = dict(scene_configs=[scene_outer_med],
                    **rest_config,
                    )

plan_med_inner = dict(scene_configs=[scene_inner_med],
                    **rest_config,
                    )

plan_easy_outer = dict(scene_configs=[scene_outer_easy],
                    **rest_config,
                    )

plan_easy_inner = dict(scene_configs=[scene_inner_easy],
                    **rest_config,
                    )
