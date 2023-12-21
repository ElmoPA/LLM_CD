import os
import json
import mlflow
import argparse
from stable_baselines3 import PPO
from utils.DMC_Gym import DMCGym
from utils.custom_environment import cmu_humanoid_run_gaps
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ld', type=str)
  parser.add_argument('--d', type=str)
  parser.add_argument('--n', type=str)
  parser.add_argument('--epn', type=str)
  args = parser.parse_args()

  timestep_file = os.path.join(args.d, 'time_step.json')
  run_id_file = os.path.join(args.d, 'mlflow_id.json')

  def make_env():
      env = cmu_humanoid_run_gaps()
      return DMCGym(env)

  num_envs = 1 
  vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

  model = PPO.load(args.ld, vec_env)

  obs = vec_env.reset()
  count = 0
  total_rw = 0
  while True:
    action, _states = model.predict(obs)
    obs, rw, dn, inf = vec_env.step(action)
    rw += total_rw
    if dn:
      break

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

  run_id = load_mlflow_run_id(run_id_file)
  mlflow_run = mlflow.start_run(run_id=run_id)
  total_timesteps = load_last_timestep(timestep_file)

  mlflow.log_metric('eval_reward', rw, step=total_timesteps)
  print(rw, total_timesteps)

  vec_env.close()