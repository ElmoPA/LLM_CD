import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from stable_baselines3 import PPO
from dm_control.composer.variation import distributions
from utils.callbacks import SummaryWriterCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
import numpy as np

env_kwargs = {
    "target_velocity": 3.0,
    "corridor_length": 100,
}

parser = argparse.ArgumentParser(description='parameters input')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--ts', type=int, default=2500000)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--load', type=str)
args = parser.parse_args()

log_dir = "./logs/"

env = cmu_humanoid_run_gaps()
env_wrap = DMCGym(env)

if args.load:
    model = PPO.load(args.load)
    model.set_env(env_wrap)
else:
    model = PPO("MlpPolicy", env_wrap,
                learning_rate=args.lr, 
                verbose=0, 
                batch_size=args.bs, 
                tensorboard_log=log_dir)

if args.load:
    print("Loaded")
    start = round((4 * model.num_timesteps/args.ts)/0.2) * 0.2
    first_iter = False
else:
    start = 1.0
    first_iter = True

total = 0.0
for i in np.arange(1.0, 4.0, 0.2):
    total += 1/i
k = args.ts/total

checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=False,
  save_vecnormalize=True,
)
callback = CallbackList([checkpoint_callback, SummaryWriterCallback()])

for i in np.arange(start, 4.0, 0.2):
    gap = np.log2(i) 
    if (gap > 0.8):
        gap_length = distributions.Uniform(0.5, gap)
    else:
        gap_length = gap
    env = cmu_humanoid_run_gaps(gap_length=gap_length)
    env_wrap = DMCGym(env)
    model.set_env(env_wrap)
    model.learn(total_timesteps=k * (1/ i), 
              tb_log_name="first_run", 
              reset_num_timesteps=first_iter,
              callback=callback)
    first_iter=False
model.save("ppo_humanoid")