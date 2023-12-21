import os
import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from utils.callbacks import MLflowCallback
from dm_control.composer.variation import distributions
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_env(env_kwargs):
    env = cmu_humanoid_run_gaps(random_state=0, **env_kwargs)
    return DMCGym(env)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='parameters input')
  parser.add_argument('--ld', type=str)
  parser.add_argument('--s', type=str)
  parser.add_argument('--ts', type=int)
  parser.add_argument('--n', type=str)
  parser.add_argument('--d', type=str)
  parser.add_argument('--g', type=float)
  args = parser.parse_args()
  env_kwargs = {
      "target_velocity": 3.0,
      "gap_length": distributions.Uniform(.5, 1.25),
      "corridor_length": 100,
  }
  if args.g:
     env_kwargs["gap_length"] = distributions.Uniform(0, args.g)
  vec_env = make_env(env_kwargs)

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
