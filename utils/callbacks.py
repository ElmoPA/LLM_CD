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

def save_total_timesteps(file_path, total_timesteps):
    with open(file_path, 'w') as f:
        json.dump({'total_timesteps': total_timesteps}, f)

def load_total_timesteps(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data['total_timesteps']
    except FileNotFoundError:
        return 0

import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir, timestep_file):
        super(TensorboardCallback, self).__init__()
        self.log_dir = log_dir
        self.timestep_file = timestep_file
        self.writer = None
        self.total_timesteps = 0

    def _on_training_start(self):
        self.total_timesteps = load_total_timesteps(self.timestep_file)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _on_step(self):
        self.total_timesteps += 1
        if self.total_timesteps % 1000 == 0:
            self.writer.add_scalar("reward", self.locals["rewards"].mean(), self.total_timesteps)
            # Add more metrics as needed

        return True

    def _on_training_end(self):
        save_total_timesteps(self.timestep_file, self.total_timesteps)
        if self.writer:
            self.writer.close()

    def _on_training_end(self):
        # Close the writer
        if self.writer:
            self.writer.close()