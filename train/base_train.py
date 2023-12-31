import os
import argparse
from utils.DMC_Gym import DMCGym
from utils.callbacks import MLflowCallback
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.custom_policy import CustomActorCriticPolicy
from dm_control.composer.variation import distributions
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import gym
from gym.envs.registration import register

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

def make_env(env_kwargs):
    env = cmu_humanoid_run_gaps(random_state=0, **env_kwargs)
    return DMCGym(env)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='parameters input')
    parser.add_argument('--d', type=str)
    parser.add_argument('--lr', type=float, default=(1e-4))
    parser.add_argument('--ts', type=int, default=100000)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--ent', type=float, default=0.0)
    parser.add_argument('--n', type=str)
    parser.add_argument('--g', type=float)
    args = parser.parse_args()
    num_envs = 1 
    env_kwargs = {
        "target_velocity": 4.0,
        "gap_length": distributions.Uniform(.5, 1.25),
        "corridor_length": 100,
    }
    if args.g:
        env_kwargs["gap_length"] = args.g
    vec_env = make_env(env_kwargs)
    net_arch = {
    'shared': [10000, 10000, 5000, 5000],  # Shared layers
    'pi': [5000, 5000, 5000, 2500, 1000, 1000, 500],  # Separate layers for the actor
    'vf': [5000, 2500, 1000, 500, 250]   # Separate layers for the critic
    }
    model = PPO(ActorCriticPolicy, vec_env,
                learning_rate=args.lr, 
                verbose=1,
                ent_coef=0.005,
                policy_kwargs={'net_arch': net_arch},
                batch_size=args.bs)
    
    model.learn(total_timesteps=args.ts,
                progress_bar=True,
                callback=MLflowCallback(args.n,
                                  os.path.join(args.d, "mlflow_id.json"),
                                  os.path.join(args.d, "time_step.json")))
    model.save(os.path.join(args.d, args.n + "0"))