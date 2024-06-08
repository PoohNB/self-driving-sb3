# semantic segmentation
from segmentation.seg_hf import HF_mask2Formermodel

# variational autoencoder
from autoencoder.vae_wrapper import VencoderWrapper,DecoderWrapper

# observer module (convert raw input to proper state)
# donb't need vae_decoder if not reconstruct the latent
from environment.tools.observer import SegVaeActHistObserver

# rewarder (object to calculate the reward for env)
from environment.tools.rewarder import RewardFromMap

# action wrapper (object to post process the action like smooth, limit range or discretize)
from environment.tools.action_wraper import LimitAction,OriginAction

# carla environment
from environment.CarlaEnv import CarlaImageEnv


def env_from_config(CONFIG,RENDER):
    assert CONFIG['env_config']['discrete_actions'] is not None and \
    (CONFIG['observer_config']['act_num'] ==1) or \
    CONFIG['env_config']['discrete_actions'] is None and \
    (CONFIG['observer_config']['act_num'] ==2)

    seg_model = HF_mask2Formermodel(**CONFIG['seg_config'])
    vae_encoder = VencoderWrapper(**CONFIG['vencoder_config'])

    if RENDER:
        vae_decoder = DecoderWrapper(**CONFIG['decoder_config'])
    else:
        vae_decoder = None

    observer = SegVaeActHistObserver(vae_encoder = vae_encoder,
                                    seg_model=seg_model,
                                    vae_decoder=vae_decoder,
                                    **CONFIG['observer_config'])

    rewarder = RewardFromMap(**CONFIG['rewarder_config'])
    # rewarder1 = RewardDummy()
    if CONFIG['actionwrapper'] is not None:
        if CONFIG['actionwrapper']['name'] == 'LimitAction':
            action_wrapper = LimitAction(**CONFIG['actionwrapper']['config'])
        else:
            print("not using action wrapper")
            action_wrapper = OriginAction()
    else:
        action_wrapper = None

    env = CarlaImageEnv(observer=observer,
                    rewarder=rewarder,
                    action_wrapper = action_wrapper,
                    activate_render = RENDER, 
                    render_raw=RENDER,
                    render_observer=RENDER,
                    **CONFIG['env_config'])
    
    return env
