from env_lib import cmu_humanoid_run_gaps
from dm_control import suite
from dm_control import viewer
from DMC_Gym import DMCGym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import CheckpointCallback

env_kwargs = {
    "target_velocity": 4.0,
    "gap_length": (.5, 1.25),
    "corridor_length": 100,
}

env = cmu_humanoid_run_gaps()
action_spec = env.action_spec()
observation_spec = env.observation_spec()
env_wrap = DMCGym(env)

log_dir = "./logs/"

model = PPO("MlpPolicy", env_wrap, verbose=1, batch_size=64, tensorboard_log="./logs/")

class SummaryWriterCallback(BaseCallback):
    def _on_training_start(self):
        self._log_freq = 10  # log every 10 calls

        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_writer = SummaryWriter('./logs/')

    def _on_step(self) -> bool:
        '''
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        '''
        if self.n_calls % self._log_freq == 0:
            self.tb_writer.add_scalar('Reward', self.locals['rewards'][0], self.num_timesteps)
        return True

model.learn(total_timesteps=50000, tb_log_name="first_run", callback=SummaryWriterCallback())
model.save("ppo_humanoid")
# Define a uniform random policy.
# for i in observation_spec.values():
#   print(i)
