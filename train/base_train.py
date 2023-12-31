import os
import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from utils.callbacks import MLflowCallback
from dm_control.composer.variation import distributions
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gym
from gym.envs.registration import register

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

def make_env():
    env = cmu_humanoid_run_gaps()
    return DMCGym(env)

if __name__ == '__main__':
    num_envs = 4 
    vec_env = SubprocVecEnv([make_env for _ in range(num_envs)])

    env_kwargs = {
        "target_velocity": 3.0,
        "gap_length": distributions.Uniform(.5, 1.25),
        "corridor_length": 100,
    }

    parser = argparse.ArgumentParser(description='parameters input')
    parser.add_argument('--d', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ts', type=int, default=100000)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--n', type=str)
    args = parser.parse_args()

    policy_kw = dict(net_arch=dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512]))
    model = PPO("MlpPolicy", vec_env,
                learning_rate=args.lr, 
                verbose=1, 
                ent_coef=0.01,
                vf_coef=1,
                policy_kwargs=policy_kw,
                batch_size=args.bs)
    
    model.learn(total_timesteps=args.ts,
                progress_bar=True,
                callback=MLflowCallback(args.n,
                                  os.path.join(args.d, "mlflow_id.json"),
                                  os.path.join(args.d, "time_step.json")))
    model.save(os.path.join(args.d, args.n + "0"))