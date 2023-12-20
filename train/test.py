import argparse
from stable_baselines3 import PPO
from utils.DMC_Gym import DMCGym
from utils.custom_environment import cmu_humanoid_run_gaps
parser = argparse.ArgumentParser()
parser.add_argument('--ld', type=str)
parser.add_argument('--n', type=str)
parser.add_argument('--ep', type=str)
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

run_id = load_mlflow_run_id(self.run_id_file)
mlflow.set_experiment(self.experiment_name)
mlflow_run = mlflow.start_run()
save_mlflow_run_id(self.run_id_file, mlflow_run.info.run_id)
vec_env.close()