import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from utils.callbacks import TensorboardCallback 
from dm_control.composer.variation import distributions
from stable_baselines3 import PPO


env_kwargs = {
    "target_velocity": 3.0,
    "gap_length": distributions.Uniform(.5, 1.25),
    "corridor_length": 100,
}

parser = argparse.ArgumentParser(description='parameters input')
parser.add_argument('--n', type=str)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--ts', type=int, default=100000)
parser.add_argument('--bs', type=int, default=1024)
args = parser.parse_args()

env = cmu_humanoid_run_gaps()
action_spec = env.action_spec()
observation_spec = env.observation_spec()
env_wrap = DMCGym(env)

policy_kw = dict(net_arch=dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512]))
model = PPO("MlpPolicy", env_wrap,
            learning_rate=args.lr, 
            verbose=1, 
            ent_coef=0.01,
            vf_coef=1,
            policy_kwargs=policy_kw,
            batch_size=args.bs)


model.learn(total_timesteps=args.ts,
            progress_bar=True,
            callback=TensorboardCallback('logs/' + args.n, 'logs/time_step.json'))
model.save("logs/" + args.n)

