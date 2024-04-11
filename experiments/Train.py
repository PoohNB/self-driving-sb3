import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Define a simple curriculum
curriculum = [
    {'max_obstacles': 2},
    {'max_obstacles': 4},
    {'max_obstacles': 6}
]

# Define the environment
env = gym.make('CustomEnvironment-v0')  # Replace 'CustomEnvironment-v0' with your environment

# Wrap the environment
env = DummyVecEnv([lambda: env])

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent using curriculum learning
for stage in curriculum:
    # Set environment parameters based on the current stage of the curriculum
    env.set_stage_parameters(**stage)
    
    # Train the agent on the current stage
    model.learn(total_timesteps=10000)  # Adjust total_timesteps as needed
    
    # Evaluate the agent's performance
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward at stage {stage}: {mean_reward}")

# Save the trained model
model.save("curriculum_trained_model")
