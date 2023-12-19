import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from utils.callbacks import TensorBoardCallback
from dm_control.composer.variation import distributions
from stable_baselines3 import PPO


env_kwargs = {
    "target_velocity": 3.0,
    "gap_length": distributions.Uniform(.5, 1.25),
    "corridor_length": 100,
}

parser = argparse.ArgumentParser(description='parameters input')
parser.add_argument('--ld', type=str)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--ts', type=int, default=100000)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--name', type=str)
args = parser.parse_args()

env = cmu_humanoid_run_gaps()
action_spec = env.action_spec()
observation_spec = env.observation_spec()
env_wrap = DMCGym(env)

log_dir = "./logs/"

if args.ld:
  model = PPO.load(args.ld)
else:
  model = PPO.load("logs/rl_model_1000000_steps.zip")
model.set_env(env_wrap)

model.learn(total_timesteps=args.ts, 
          progress_bar=True,
          callback=TensorBoardCallback())

model.save("logs/" + args.name)
