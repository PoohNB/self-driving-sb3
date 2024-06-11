import optuna
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def optimize_ppo(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_int('n_steps', 16, 2048)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-2)
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 0.9)
    max_grad_norm = trial.suggest_uniform('max_grad_norm', 0.3, 0.9)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)

    env = gym.make('CartPole-v1')

    model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, 
                gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef,
                vf_coef=vf_coef, max_grad_norm=max_grad_norm, n_epochs=n_epochs, verbose=0)
    
    model.learn(total_timesteps=10000)
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(optimize_ppo, n_trials=100)
