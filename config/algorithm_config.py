from utils import exp_schedule
import torch as th
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC,PPO,DDPG,DQN
from sb3_contrib import RecurrentPPO

available_AlgorithmRL = {
    "SAC":SAC,
    "PPO":PPO,
    "DDPG":DDPG,
    "RecurrentPPO":RecurrentPPO,
    "DQN":DQN
}


SAC1 = dict(policy = "MlpPolicy",
        learning_rate=exp_schedule(1e-4, 5e-7, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3,
                            net_arch=[500, 300]),
    )

SAC1_con = dict(policy = "MlpPolicy",
        learning_rate= 5e-7,
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3,
                            net_arch=[500, 300],
                            use_sde= True),
    )



DDPG1 = dict(policy = "MlpPolicy",
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=exp_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    )

PPO1 = dict(policy = "MlpPolicy",
        learning_rate=exp_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=dict(pi=[500, 300], vf=[500, 300])))

PPO2 = dict(policy = "MlpPolicy",
        learning_rate=exp_schedule(1e-4, 1e-6, 2),
        gamma=0.995,
        gae_lambda=0.9,
        max_grad_norm=0.5,
        clip_range=0.2,
        ent_coef=0.046,
        vf_coef=0.85,
        n_epochs=10,
        n_steps=1024,
        batch_size=256,
        policy_kwargs=dict(activation_fn=th.nn.ELU,
                           net_arch=dict(pi=[64, 64,64], vf=[64, 64,64])))

RNNPPO1 = dict(policy = "MlpLstmPolicy",
        learning_rate=exp_schedule(1e-4, 1e-6, 2),
        ent_coef=0.045,)


# custom_params = {
#     'n_steps': 2048,
#     'batch_size': 64,
#     'n_epochs': 10,
#     'gamma': 0.99,
#     'learning_rate': 3e-4,
#     'ent_coef': 0.01,
#     'clip_range': 0.2,
#     'gae_lambda': 0.95,
#     'max_grad_norm': 0.5,
#     'vf_coef': 0.5,
#     'policy_kwargs': {
#         'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}],
#         'lstm_hidden_size': 256,
#         'n_lstm_layers': 2,
#         'shared_lstm': False,
#         'enable_critic_lstm': True,
#     }
# }

