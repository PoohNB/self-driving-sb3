from config.seg_config import *
from config.vae import *
seg_and_vae = dict(seg_model_config=fbm2f_fp16_1280,
                  vae_encoder_config=vencoder32,)

observer_con_manv = dict(name="SegVaeActManvHistObserver",
                 config=dict(**seg_and_vae,
                            num_img_input = 1,
                            act_num=2,
                            maneuver_num=1,
                            hist_len = 12,
                            skip_frame=0))

observer_discrete_manv = dict(name="SegVaeActManvHistObserver",
                 config=dict(**seg_and_vae,
                            num_img_input = 1,
                            act_num=1,
                            maneuver_num=1,
                            hist_len = 12,
                            skip_frame=0))

observer_raw = dict(name="RawObserver",
                 config=dict(image_size = (720,1280,3)))




