import numpy as np
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
class SummaryWriterCallback(BaseCallback):
    def _on_training_start(self):
        self._log_freq = 50  # log every 10 calls

        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        output_formats = self.logger.output_formats

        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        self.overall = 0.0

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        self.overall += self.locals['rewards']
        
        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar('Average Return', self.overall/self.n_calls, self.n_calls)
            self.tb_formatter.writer.add_scalar('Return', self.locals['rewards'], self.n_calls)
        return True

class CurrentCheckCallback(BaseCallback):
    def _on_training_start(self):
        self._log_freq = 50  # log every 10 calls

        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        output_formats = self.logger.output_formats

        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        self.overall = 0.0

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        self.overall += self.locals['rewards']
        
        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar('Average Return', self.overall/self.n_calls, self.n_calls)
            self.tb_formatter.writer.add_scalar('Return', self.locals['rewards'], self.n_calls)
        return True

import json
import mlflow

def save_last_timestep(file_path, total_timesteps):
    with open(file_path, 'w') as f:
        json.dump({'total_timesteps': total_timesteps}, f)

def load_last_timestep(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['total_timesteps']
    except FileNotFoundError:
        return 0

def save_mlflow_run_id(file_path, run_id):
    with open(file_path, 'w') as f:
        json.dump({'run_id': run_id}, f)

def load_mlflow_run_id(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['run_id']
    except FileNotFoundError:
        return None

class MLflowCallback(BaseCallback):
    def __init__(self, experiment_name, run_id_file, timestep_file):
        super().__init__()
        self.experiment_name = experiment_name
        self.run_id_file = run_id_file
        self.timestep_file = timestep_file
        self.total_timesteps = 0
        self.check_freq = 1000
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_lengths = []
        self.current_episode_length = 0

    def _on_training_start(self):
        self.total_timesteps = load_last_timestep(self.timestep_file)

        run_id = load_mlflow_run_id(self.run_id_file)
        if run_id is not None:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.set_experiment(self.experiment_name)
            mlflow_run = mlflow.start_run()
            save_mlflow_run_id(self.run_id_file, mlflow_run.info.run_id)

    def _on_step(self):
        self.total_timesteps += 1
        reward = self.locals['rewards'][0]  # Assuming a single environment
        done = self.locals['dones'][0]

        # Update current episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1

        # Check for episode completion
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # Log the episode stats
            mlflow.log_metric('episode_reward', self.episode_rewards[-1], step=self.total_timesteps)
            mlflow.log_metric('episode_length', self.episode_lengths[-1], step=self.total_timesteps)
        return True

    def _on_training_end(self):
        save_last_timestep(self.timestep_file, self.total_timesteps)
        mlflow.end_run()