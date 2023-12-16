import zipfile
import os
from env_lib import cmu_humanoid_run_gaps
from dm_control import suite
from dm_control import viewer
from DMC_Gym import DMCGym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

video_folder = "logs/videos/"
video_length = 1000
env = cmu_humanoid_run_gaps()
action_spec = env.action_spec()
observation_spec = env.observation_spec()
env_wrap = DMCGym(env)
print(check_env(env_wrap))

model = PPO.load("ppo_humanoid")
model.set_env(env_wrap)

vec_env = model.get_env()
vec_env = VecVideoRecorder(vec_env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"random-agent")
obs = vec_env.reset()
for _ in range(video_length + 1):
  action, _states = model.predict(obs)
  obs, _, _, _ = vec_env.step(action)

vec_env.close()
