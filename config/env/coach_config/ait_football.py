ait_football_spawn = [
             dict(Location=(-573.5,-223.5,0.2),Rotation=(0,270,0)),
             dict(Location=(-490.1,-369.2,0.2),Rotation=(0,0,0)),
             dict(Location=(-347.4,-273.5,0.2),Rotation=(0,90,0)),
             dict(Location=(-361,-142.9,0.2),Rotation=(0,180,0)),
             dict(Location=(-561.1,-149.3,0.2),Rotation=(0,182,0))
             
             ]

ait_fb_spawn_outer = [
             dict(Location=(-573.5,-223.5,0.2),Rotation=(0,270,0)),
             dict(Location=(-490.1,-369.2,0.2),Rotation=(0,0,0)),
             dict(Location=(-347.4,-273.5,0.2),Rotation=(0,90,0)),
             dict(Location=(-361,-142.9,0.2),Rotation=(0,180,0)),
             dict(Location=(-561.1,-149.3,0.2),Rotation=(0,182,0))
             
             ]

ait_fb_spawn_inner= [
             dict(Location=(-480.0,-366.0,0.2),Rotation=(0,180,0)),
             dict(Location=(-570.0,-280.0,0.2),Rotation=(0,90,0)),
             dict(Location=(-518.0,-151.0,0.2),Rotation=(0,0,0)),
             dict(Location=(-351.4,-271.0,0.2),Rotation=(0,270,0)),             
             ]

junction =[(-576.6548461914062, -148.419189453125),
            (-570.05419921875, -366.15301513671875),
            (-347.0089111328125, -364.6258544921875),
            (-350.5668029785156, -143.1997528076172)]


call_rad = (15,40)
# -1,0,1 => left,forward,right
cmd_points_inner = dict( name = "DirectionCmd",
                         cmd_configs=[dict(loc=(-576.6548461914062, -148.419189453125),
                                   call_rad = call_rad,
                                   cmd = -1),
                                dict(loc=(-570.05419921875, -366.15301513671875),
                                   call_rad = call_rad,
                                   cmd = -1),
                                dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = -1),
                                dict(loc=(-350.5668029785156, -143.1997528076172),
                                   call_rad = call_rad,
                                   cmd = -1),
                                   ])

cmd_points_outer = dict( name = "DirectionCmd",
                        cmd_configs=[dict(loc=(-576.6548461914062, -148.419189453125),
                                   call_rad = call_rad,
                                   cmd = 1),
                                dict(loc=(-570.05419921875, -366.15301513671875),
                                   call_rad = call_rad,
                                   cmd = 1),
                                dict(loc=(-347.0089111328125, -364.6258544921875),
                                   call_rad = call_rad,
                                   cmd = 1),
                                dict(loc=(-350.5668029785156, -143.1997528076172),
                                   call_rad = call_rad,
                                   cmd = 1),
                                   ])

# ait_fb_car_outer = dict(available_loc = [dict(Location=(-423.0,-368.1,0.1),Rotation=(0,1,0)),
#                                          dict(Location=(-506.0,-369.2,0.1),Rotation=(0,0,0)),
#                                          dict(Location=(-384.2,-367.3,0.1),Rotation=(0,1,0)),
#                                          dict(Location=(-346.1,-344.7,0.1),Rotation=(0,93,0)),
#                                          dict(Location=(-347.7,-280.4,0.1),Rotation=(0,92,0)),
#                                          dict(Location=(-348.1,-197.5,0.1),Rotation=(0,90,0)),
#                                          dict(Location=(-385.7,-143.1,0.1),Rotation=(0,180,0)),
#                                          dict(Location=(-515.4,-148.4,0.1),Rotation=(0,180,0)),
#                                          dict(Location=(-573.5,-187.1,0.1),Rotation=(0,270,0)),
#                                          dict(Location=(-572.3,-328.0,0.1),Rotation=(0,271,0)),
#                                          ],
#                         density =0.8)

ait_fb_car_outer = dict(available_loc = [(-538.1447101510797, -366.44095049198035),
                                        (-441.5904279579548, -365.62663726866487),
                                        (-350.1546860256701, -342.47687563440957),
                                        (-350.96899924898565, -288.8485333560594),
                                        (-351.8996429327748, -166.70154985873265),
                                        (-444.8476808512168, -148.3213371038968),
                                        (-551.173721724128, -151.81125091810614),
                                        (-570.0192563208584, -194.03920807003914),
                                        (-569.0886126370692, -299.6672661801084),
                                        (-568.3906298742273, -351.55065155135384)],
                         values = 6,
                         on_road_ratio = 1
                         )
                         

ait_fb_ped_outer = dict(available_loc = [(-569.32, -172.87),
                                        (-479.05, -150.53),
                                        (-383.77, -145.76),
                                        (-575.84, -282.22),
                                        (-499.17, -145.65),
                                        (-331.77, -365.98)],
                        values=4)

ait_fb_car_inner = dict(available_loc = [dict(Location=(),Rotation=()),
                                         dict(Location=(),Rotation=())],
                        values =5,
                        on_road_ratio=0.8
                        )

ait_fb_ped_inner = dict(available_loc = [dict(Location=(),Rotation=()),
                                            dict(Location=(),Rotation=())],
                        values=4)

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


scene_config_outer = dict(spawn_points = ait_fb_spawn_outer,
                          cmd_config = cmd_points_outer,
                          car_obsc = ait_fb_car_outer,
                          ped_obsc = ait_fb_ped_outer,
                          rewarder_config = dict(name="RewardPath",
                                        config = dict(mask_path = "environment/rewardmask/ait_map/ait_fb_outer.png")))

scene_config_inner = dict(spawn_points = ait_fb_spawn_inner,
                          cmd_config = cmd_points_inner,
                          car_obsc = ait_fb_car_inner,
                          ped_obsc = ait_fb_ped_inner,
                          rewarder_config = dict(name="RewardPath",
                                        config = dict(mask_path = "environment/rewardmask/ait_map/ait_fb_inner.png")))

ait_fb_scene = dict(scene_configs=[scene_config_inner,scene_config_outer],
                    parking_area=parking_available,
                    ped_area=people_area
                    )

ait_fb_scene_outer_only = dict(scene_configs=[scene_config_outer],
                    parking_area=parking_available,
                    ped_area=people_area
                    )
