from utils import lr_schedule
import torch as th
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise


SAC1 = dict(
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[500, 300]),
    )


DDPG1 = dict(
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    )

PPO1 = dict(
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])]))


RNNPPO1 = dict()

