

def PPO_objective(trial: optuna.Trial) -> float:
    global env

    time.sleep(random.random() * 16)

    step_mul = trial.suggest_categorical("step_mul", [4, 8, 16, 32, 64])

    sampled_hyperparams = sample_ppo_params(trial)

    path = f"optuna_trials/PPO/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    env = Monitor(env)
    model = PPO("MlpPolicy", env=env, seed=None, verbose=0, tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=3, eval_freq=2500, deterministic=False, callback_after_eval=stop_callback
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

def RNNPPO_objective(trial: optuna.Trial) -> float:
    global env

    time.sleep(random.random() * 16)

    step_mul = trial.suggest_categorical("step_mul", [4, 8, 16, 32, 64])

    sampled_hyperparams = sample_rnnppo_params(trial)

    path = f"optuna_trials/PPO/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    env = Monitor(env)
    model = RecurrentPPO("MlpLstmPolicy", env=env, seed=None, verbose=0, tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=3, eval_freq=2500, deterministic=False, callback_after_eval=stop_callback
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