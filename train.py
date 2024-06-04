
# initial segmodel and encoder
from segmentation.seg_hf import HF_mask2Formermodel
from autoencoder.vae_wrapper import VencoderWrapper,DecoderWrapper
from environment.tools.observer import SegVaeActHistObserver
from environment.tools.rewarder import RewardFromMap
from environment.tools.action_wraper import LimitAction,OriginAction
from environment.CarlaEnv import CarlaImageEnv

from config.env.camera import front_cam
from config.env.spawn_points import ait_football_spawn
from config.env.reward_config import ait_map
from config.env.env_config import ait_football_env
from config.env.action_config import limit_action
import time

modelrepo = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
mapping = {13:1,7:1,23:2,24:2,52:3,55:3,57:3,20:4,21:4,22:4,19:4}
seg_model = HF_mask2Formermodel(model_repo=modelrepo,fp16=True,label_mapping=mapping,crop=(512,1024))
#  # 0 is back ground
# seg_model.apply_label_mapping(mapping)
vae_encoder = VencoderWrapper(model_path="autoencoder/model/vae32/best/var_encoder_model.pth",latent_dims=32)
vae_decoder = DecoderWrapper(model_path="autoencoder/model/vae32/best/decoder_model.pth",latent_dims=32)

observer1 = SegVaeActHistObserver(vae_encoder = vae_encoder,
                                  seg_model=seg_model,
                                  vae_decoder=vae_decoder,
                                  latent_space=32,
                                  hist_len = 8,
                                  skip_frame=0)

rewarder1 = RewardFromMap(route_config=ait_map)

action_wrapper = LimitAction(limit_action,stable_steer=False)

env = CarlaImageEnv(observer=observer1,
                rewarder=rewarder1,   
                car_spawn=ait_football_spawn,
                action_wrapper = OriginAction(), 
                env_config =ait_football_env,
                cam_config_list=[front_cam], 
                discrete_actions = None,
                activate_render = True,
                seed=2024,
                render_raw=True,
                render_seg=True,
                render_reconst=True)

try:
    observation = env.reset()
    done = False


    act = [0.0,0.6]
    while True:
        observation, reward, done, _ = env.step(act)
        if done :
            break

except Exception as e:
    print(e)

finally:

    env.close()