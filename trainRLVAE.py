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

from config.trainRL_config import RL1

from utils import HParamCallback,TensorboardCallback

from stable_baselines3 import SAC,PPO,DDPG
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

available_policy = {"SAC":SAC,"PPO":PPO,"DDPG":DDPG,"RecurrentPPO":RecurrentPPO}

CONFIG = RL1
RENDER = True
LOG_DIR = "runs/RL"

assert CONFIG['env_config']['discrete_actions'] is not None and \
    (CONFIG['observer_config']['act_num'] ==1) or \
    CONFIG['env_config']['discrete_actions'] is None and \
    (CONFIG['observer_config']['act_num'] ==2)

seg_model = HF_mask2Formermodel(**CONFIG['seg_config'])
vae_encoder = VencoderWrapper(**CONFIG['vae_config'])

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

if CONFIG['actionwrapper']['name'] == 'LimitAction':
    action_wrapper = LimitAction(**CONFIG['actionwrapper']['config'])
else:
    action_wrapper = OriginAction()

env = CarlaImageEnv(observer=observer,
                rewarder=rewarder,
                action_wrapper = action_wrapper,
                activate_render = RENDER, 
                render_raw=RENDER,
                render_observer=RENDER,
                **CONFIG['env_config'])

Policy = available_policy[CONFIG["algorithm"]["policy"]]
model = Policy('MlpPolicy', 
               env, verbose=1, 
               seed=CONFIG['algorithm']['seed'], 
               tensorboard_log=LOG_DIR, device='cuda',
               **CONFIG['algorithm']['model_config'])

model.learn(total_timesteps=CONFIG['train_config']['total_timesteps'],
            callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
                save_freq=CONFIG['train_config']['total_timesteps'] // CONFIG['train_config']["num_checkpoints"],
                save_path=CONFIG['train_config']['save_path'],
                name_prefix="model")], reset_num_timesteps=False)