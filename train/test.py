import json
import mlflow
import argparse
from stable_baselines3 import PPO
from utils.DMC_Gym import DMCGym
from utils.custom_environment import cmu_humanoid_run_gaps
parser = argparse.ArgumentParser()
parser.add_argument('--ld', type=str)
parser.add_argument('--n', type=str)
parser.add_argument('--epn', type=str)
parser.add_argument('--r', type=str)
args = parser.parse_args()

env = cmu_humanoid_run_gaps()
env_wrap = DMCGym(env)

model = PPO.load(args.ld)
model.set_env(env_wrap)

vec_env = model.get_env()

obs = vec_env.reset()
count = 0
total_rw = 0
while True:
  action, _states = model.predict(obs)
  obs, rw, dn, inf = vec_env.step(action)
  rw += total_rw
  if dn:
    break

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

run_id = load_mlflow_run_id(args.r)
mlflow.set_experiment(args.epn)
mlflow_run = mlflow.start_run()
save_mlflow_run_id(args.r, mlflow_run.info.run_id)
vec_env.close()