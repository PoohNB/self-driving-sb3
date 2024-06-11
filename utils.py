# ==============================================================================
# -- copied and modified from Alberto Maté Angulo --
# ==============================================================================


import cv2
import math
import json
import pickle

import gym
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import numpy as np
import os
import re


def create_policy_paths(save_path,log_dir,policy_name):

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    pattern = re.compile(rf"{re.escape(policy_name)}_\d+$")

    entries = os.listdir(log_dir)
    num_run_policy = max([int(entry.split("_")[-1]) for entry in entries if os.path.isdir(os.path.join(log_dir, entry)) and pattern.match(entry)]+[0])
    # num_run_policy = len([entry for entry in entries if os.path.isdir(os.path.join(log_dir, entry)) and policy_name in entry])
    name = policy_name+'_'+str(num_run_policy+1)
    new_save_path = os.path.join(save_path,name)
    new_log_dir = os.path.join(log_dir,name)
    return new_save_path,new_log_dir

def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def write_json(data, path):
    def serialize_value(value):
        """Serialize a value to a string if it's not already an acceptable type."""
        
        if isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif callable(value):
            return value.__str__()
        else:
            return value

    hparam_dict = {k: serialize_value(v) for k, v in data.items()}
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(hparam_dict, f, indent=4)




class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename, fourcc, int(fps), (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


class HParamCallback(BaseCallback):
    def __init__(self, config):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard. - modified
        """
        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        

        def serialize_value(value):
            """Serialize a value to a string if it's not already an acceptable type."""
            if isinstance(value, (int, float, str, bool)):
                return value
            elif isinstance(value, dict):
                return str({k: serialize_value(v) for k, v in value.items()})
            elif callable(value):
                return value.__str__()
            else:
                return str(value)

        hparam_dict = {k: serialize_value(v) for k, v in self.config.items()}
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.locals['dones'][0]:
            self.logger.record("custom/total_reward", self.locals['infos'][0]['total_reward'])
            self.logger.record("custom/total_distance", self.locals['infos'][0]['total_distance'])
            # self.logger.record("custom/avg_center_dev", self.locals['infos'][0]['avg_center_dev'])
            self.logger.record("custom/avg_speed", self.locals['infos'][0]['avg_speed'])
            self.logger.record("custom/mean_reward", self.locals['infos'][0]['mean_reward'])
            self.logger.dump(self.num_timesteps)
        return True

class VideoRecorderCallback(BaseCallback):
    def __init__(self, video_path, frame_size, video_length=-1, fps=30, skip_frame=1, verbose=0):
        super().__init__(verbose)
        self.video_recorder = VideoRecorder(video_path, frame_size, fps)
        self.max_length = video_length
        self.skip_frame = skip_frame

    def _on_step(self) -> bool:
        # Add frame to video
        if self.max_length != -1 and self.num_timesteps > self.max_length:
            self.video_recorder.release()
            return False
        # Skip every 4 frames to reduce video size
        if self.num_timesteps % self.skip_frame != 0:
            return True
        display = self.training_env.unwrapped.envs[0].env.pygamectrl.display
        frame = np.array(pygame.surfarray.array3d(display), dtype=np.uint8).transpose([1, 0, 2])

        self.video_recorder.add_frame(frame)
        return True

    def _on_training_end(self) -> None:
        self.video_recorder.release()


def lr_schedule(initial_value: float, end_value: float, rate: float):
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

    func.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"
    lr_schedule.__str__ = lambda: f"lr_schedule({initial_value}, {end_value}, {rate})"

    return func


class SaveOnBestTrainingRewardCallback(BaseCallback):

    """
        this function for save model every save_freq, also check and save model if model have best mean reward every check_freq 
    
    """

    def __init__(self, check_freq: int,save_freq:int ,save_path:str,log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:   
                  print("Saving new best model to {}".format(self.save_path))
                model_path = os.path.join(self.save_path, 'best_model')
                self.model.save(model_path)
                
        if self.n_calls %self.save_freq ==0:
          if self.verbose > 0:
            print("Saving checkpoint model to {}".format(self.save_path))
          model_path = os.path.join(self.save_path, 'model_{}'.format(self.n_calls))
          self.model.save(model_path)
                 

        return True


class HistoryWrapperObsDict(gym.Wrapper):
    # History Wrapper from rl-baselines3-zoo
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/10de3a8804b14b4ea605b487ae7d8117c52901c4/rl_zoo3/wrappers.py
    """
    History Wrapper for dict observation.
    :param env:
    :param horizon: Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 2, obs_key: str = 'vae_latent') -> object:
        self.obs_key = obs_key
        assert isinstance(env.observation_space.spaces[obs_key], gym.spaces.Box)
        print("Wrapping the env with HistoryWrapperObsDict.")
        wrapped_obs_space = env.observation_space.spaces[self.obs_key]
        wrapped_action_space = env.action_space

        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space.spaces[obs_key] = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super().__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs_dict = self.env.reset()
        obs = obs_dict[self.obs_key]
        self.obs_history[..., -obs.shape[-1]:] = obs

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = obs_dict[self.obs_key]
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action

        obs_dict[self.obs_key] = self._create_obs_from_history()

        return obs_dict, reward, done, info


class FrameSkip(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        print("Wrapping the env with FrameSkip.")
        self._skip = skip

    def step(self, action: np.ndarray):
        """
        Step the environment with the given action
        Repeat action, sum reward.
        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()


def parse_wrapper_class(wrapper_class_str: str):
    """
    Parse a string to a wrapper class.

    :param wrapper_class_str: (str) The string to parse.
    :return: (type) The wrapper class and its parameters.
    """
    wrap_class, wrap_params = wrapper_class_str.split("_", 1)
    wrap_params = wrap_params.split("_")
    wrap_params = [int(param) if param.isnumeric() else param for param in wrap_params]

    if wrap_class == "HistoryWrapperObsDict":
        return HistoryWrapperObsDict, wrap_params
    elif wrap_class == "FrameSkip":
        return FrameSkip, wrap_params