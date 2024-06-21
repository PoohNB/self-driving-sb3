# semantic segmentation
from segmentation.seg_hf import HF_mask2Formermodel

# variational autoencoder
from autoencoder.vae_wrapper import VencoderWrapper,DecoderWrapper

# observer module (convert raw input to proper state)
# donb't need vae_decoder if not reconstruct the latent
from environment.tools.observer import observer_type

# action wrapper (object to post process the action like smooth, limit range or discretize)
from environment.tools.action_wraper import action_wrapper_type

# carla environment
from environment.CarlaEnv import CarlaImageEnv
from environment.CtrlEnv import ManualCtrlEnv

def init_component(config):
    assert config['env_config']['discrete_actions'] is not None and \
    (config['observer_config']['config']['act_num'] ==1) or \
    config['env_config']['discrete_actions'] is None and \
    (config['observer_config']['config']['act_num'] >=2)

    seg_model = HF_mask2Formermodel(**config['seg_config'])
    vae_encoder = VencoderWrapper(**config['vencoder_config'])

    if (config['decoder_config'] is not None):
        vae_decoder = DecoderWrapper(**config['decoder_config'])
    else:
        vae_decoder = None

    observer = observer_type[config['observer_config']['name']](vae_encoder = vae_encoder,
                                                                seg_model=seg_model,
                                                                vae_decoder=vae_decoder,
                                                                **config['observer_config']['config'])
  

    action_wrapper = action_wrapper_type[config['actionwrapper']['name']](**config['actionwrapper']['config'])

    return observer,action_wrapper



def env_from_config(config,RENDER):
    observer,action_wrapper =init_component(config)

    env = CarlaImageEnv(observer=observer,
                    action_wrapper = action_wrapper,
                    activate_render = RENDER, 
                    render_raw=RENDER,
                    render_observer=RENDER,
                    **config['env_config'])
    
    return env

def manualctrlenv_from_config(config,sync):

    observer,action_wrapper =init_component(config)
    env = ManualCtrlEnv(observer=observer,
                    action_wrapper = action_wrapper,
                    **config['env_config'])
    
    return env
    


