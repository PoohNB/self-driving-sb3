# for speed 8 km/hr
reward_mask_base = dict(reward_scale=8,
                       max_velo=2.22,
                       max_angular_velo=30.5,
                       step_time= 0.2,
                       max_steer=0.6,
                       minimum_distance = 0.015,
                       mid_steer_range = 0.1,
                       out_of_road_count_limit = 20,
                       staystill_limit=25)

admin2aic = dict(name="RewardMaskPathV1",
                    config = dict(mask_path = "environment/rewardmask/ait_map/admin2aic.png",
                                  end_point = (-243.94, -369.58)))

aic2admin = dict(name="RewardMaskPathV1",
                    config = dict(mask_path = "environment/rewardmask/ait_map/aic2admin.png",
                                  end_point = (2.91, -98.42)))

ait_fb_outer = dict(name="RewardMaskPathV1",
                    config = dict(mask_path = "environment/rewardmask/ait_map/ait_fb_outer.png",
                                  end_point = None))

ait_fb_inner = dict(name="RewardMaskPathV1",
                    config = dict(mask_path = "environment/rewardmask/ait_map/ait_fb_inner.png",
                                  end_point = None))


