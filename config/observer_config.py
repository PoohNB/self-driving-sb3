
observer_con = dict(name="SegVaeActHistObserver",
                 config=dict(num_img_input = 1,
                            act_num=2,
                            hist_len = 12,
                            skip_frame=0))


observer_con_manv = dict(name="SegVaeActHistManvObserver",
                 config=dict(num_img_input = 1,
                            act_num=2,
                            maneuver_num=1,
                            hist_len = 12,
                            skip_frame=0))

observer_discrete = dict(name="SegVaeActHistObserver",
                 config=dict(num_img_input = 1,
                            act_num=1,
                            hist_len = 12,
                            skip_frame=0))

observer_con_no_hist = dict(name="SegVaeActObserver",
                 config=dict(num_img_input = 1,
                            act_num=2,))

observer_discrete_no_hist = dict(name="SegVaeActObserver",
                 config=dict(num_img_input = 1,
                            act_num=1,))



