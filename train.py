# semantic segmentation
from segmentation.seg_hf import HF_mask2Formermodel
from config.seg_config import mask2former_labelmap

# variational autoencoder
from autoencoder.vae_wrapper import VencoderWrapper,DecoderWrapper

# observer module (convert raw input to proper state)
# donb't need vae_decoder if not reconstruct the latent
from environment.tools.observer import SegVaeActHistObserver

# rewarder (object to calculate the reward for env)
from environment.tools.rewarder import RewardFromMap,RewardDummy

# action wrapper (object to post process the action like smooth, limit range or discretize)
from environment.tools.action_wraper import LimitAction,OriginAction
from config.env.action_config import limit_action

# carla environment
from environment.CarlaEnv import CarlaImageEnv

from config.env.camera import front_cam,left_cam
from config.env.spawn_points import ait_football_spawn
from config.env.env_config import ait_football_env

modelrepo = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
seg_model = HF_mask2Formermodel(model_repo=modelrepo,fp16=True,label_mapping=mask2former_labelmap,crop=(512,1024))

vae_encoder = VencoderWrapper(model_path="autoencoder/model/vae32/best/var_encoder_model.pth",latent_dims=32)
vae_decoder = DecoderWrapper(model_path="autoencoder/model/vae32/best/decoder_model.pth",latent_dims=32)

observer1 = SegVaeActHistObserver(vae_encoder = vae_encoder,
                                  seg_model=seg_model,
                                  vae_decoder=vae_decoder,
                                  latent_space=32,
                                  num_img_input = 1,
                                  act_num=2,
                                  hist_len = 8,
                                  skip_frame=0)

rewarder1 = RewardFromMap(mask_path="environment/rewardmask/ait_map/ait_football.png")
# rewarder1 = RewardDummy()

action_wrapper = LimitAction(limit_action,stable_steer=True)
# action_wrapper = OriginAction()

env = CarlaImageEnv(observer=observer1,
                rewarder=rewarder1,   
                car_spawn=ait_football_spawn,
                spawn_mode='static',
                action_wrapper = action_wrapper, 
                env_config =ait_football_env,
                cam_config_list=[front_cam], 
                discrete_actions = None,
                activate_render = True,
                seed=2024,
                render_raw=True,
                render_seg=True,
                render_reconst=True,
                rand_weather=False)


# try:
#     for _ in range(3):
#         observation = env.reset()
#         done = False

#         act = [1,0.5]
#         while True:
#             observation, reward, done, _ = env.step(act)
#             if done :
#                 break

# except Exception as e:
#     print(e)

# finally:

#     env.close()

