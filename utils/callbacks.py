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