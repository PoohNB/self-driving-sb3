from typing import Dict, Any, Union, Callable

import optuna
from torch import nn
import math


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    func.__str__ = lambda: f"linear_schedule({initial_value})"
    linear_schedule.__str__ = lambda: f"linear_schedule({initial_value})"

    return func

def Exponential_decay(initial_value: float, end_value: float, rate: float):
    """
    Learning rate schedule:
        Exponential decay by factors of 10 from initial_value to end_value.

    :param initial_value: Initial learning rate.
    :param rate: Exponential rate of decay. High values mean fast early drop in LR
    :param end_value: The final value of the learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: A float value between 0 and 1 that represents the remaining progress.
        :return: The current learning rate.
        """
        if progress_remaining <= 0:
            return end_value

        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    func.__str__ = lambda: f"Exponential_decay({initial_value}, {end_value}, {rate})"
    Exponential_decay.__str__ = lambda: f"Exponential_decay({initial_value}, {end_value}, {rate})"

    return func


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size",  [ 256, 512, 1024])
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant','exponential'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.0001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [ 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [ 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 0.6, 0.7, 0.8, 0.9, 1])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)

    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', [ 'relu', 'elu', 'leaky_relu','tanh'])
    activation_fn = trial.suggest_categorical('activation_fn', [ 'relu', 'elu'])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)
    elif lr_schedule == "exponential":
        learning_rate = Exponential_decay(learning_rate,1e-6,2)

    # Independent networks usually work best
    # when not working with images
    """
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    net_arch = {
        "tiny": [dict(pi=[8, 8], vf=[8, 8])],
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]
    """
    # net_arch_width = trial.suggest_categorical("net_arch_width", [ 64, 128, 256, 512])
    # net_arch_depth = trial.suggest_int("net_arch_depth", 1, 3)
    # net_arch = [dict(pi=[net_arch_width] * net_arch_depth, vf=[net_arch_width] * net_arch_depth)]
    # if want static network
    net_arch = [dict(pi=[500,300], vf=[500,300])]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]
    activation_fn = { "relu": nn.ReLU, "elu": nn.ELU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
