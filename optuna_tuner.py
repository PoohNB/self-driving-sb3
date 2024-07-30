
import os
import pickle as pkl
import random
import sys
import time
from pprint import pprint

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from optuna_utils.sample_params.ppo import sample_ppo_params
from optuna_utils.trial_eval_callback import TrialEvalCallback
from environment.loader import env_from_config
from stable_baselines3.common.vec_env import DummyVecEnv
from config.Config_loader import get_env_config
from utils import write_json


CONFIG = get_env_config(action_type="continuous")
env = env_from_config(CONFIG,True)
# env = DummyVecEnv([lambda: env])

FLAGS = flags.FLAGS
FLAGS(sys.argv)


study_path = "optuna_trials/PPO"

os.makedirs(study_path, exist_ok=True)


with open(f"{study_path}/env_config.pkl",'wb+') as f:
    pkl.dump(CONFIG,f)


def objective(trial: optuna.Trial) -> float:
    global env

    time.sleep(random.random() * 16)

    # step_mul = trial.suggest_categorical("step_mul", [4, 8, 16, 32, 64])

    sampled_hyperparams = sample_ppo_params(trial)

    path = f"{study_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    env = Monitor(env)
    model = PPO("MlpPolicy", env=env, seed=2024, verbose=0, tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=7, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=1, eval_freq=2500, deterministic=False, callback_after_eval=stop_callback
    )

    params = sampled_hyperparams
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(50000, callback=eval_callback)
    except (AssertionError, ValueError) as e:
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


if __name__ == "__main__":

    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=1, n_trials=8)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)


    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)