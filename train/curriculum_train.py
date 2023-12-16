import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from stable_baselines3 import PPO
from utils.callbacks import SummaryWriterCallback
import numpy as np

env_kwargs = {
    "target_velocity": 4.0,
    "corridor_length": 100,
}

parser = argparse.ArgumentParser(description='parameters input')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--ts', type=int, default=2500000)
parser.add_argument('--bs', type=int, default=1024)
args = parser.parse_args()

log_dir = "./logs/"

env = cmu_humanoid_run_gaps()
env_wrap = DMCGym(env)

model = PPO("MlpPolicy", env_wrap,
            learning_rate=args.lr, 
            verbose=0, 
            batch_size=args.bs, 
            tensorboard_log=log_dir)

total = 0.0
for i in np.arange(1.0, 4.0, 0.2):
    total += 1/i
k = args.ts/total
first_iter = True

for i in np.arange(1.0, 4.0, 0.2):
    gap = np.log2(i) 
    env = cmu_humanoid_run_gaps(gap_length=gap)
    env_wrap = DMCGym(env)
    model.set_env(env_wrap)
    model.learn(total_timesteps=k * (1/ i), 
              tb_log_name="first_run", 
              reset_num_timesteps=first_iter,
              callback=SummaryWriterCallback())
    
    first_iter=False
model.save("ppo_humanoid")