import os
import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from utils.callbacks import MLflowCallback
from dm_control.composer.variation import distributions
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    env = cmu_humanoid_run_gaps()
    return DMCGym(env)

if __name__ == '__main__':
  num_envs = 8
  vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
  env_kwargs = {
      "target_velocity": 3.0,
      "gap_length": distributions.Uniform(.5, 1.25),
      "corridor_length": 100,
  }

  parser = argparse.ArgumentParser(description='parameters input')
  parser.add_argument('--ld', type=str)
  parser.add_argument('--s', type=str)
  parser.add_argument('--ts', type=int)
  parser.add_argument('--n', type=str)
  parser.add_argument('--d', type=str)
  args = parser.parse_args()

  log_dir = "./logs/"

  if args.ld:
    model = PPO.load(args.ld)
  else:
    raise Exception("No load file specified")

  model.set_env(vec_env)

  model.learn(total_timesteps=args.ts, 
            progress_bar=True,
            callback=MLflowCallback(args.n,
                                    os.path.join(args.d, "mlflow_id.json"),
                                    os.path.join(args.d, "time_step.json")))

  model.save(args.s)
