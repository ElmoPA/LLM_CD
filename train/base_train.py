import argparse
from utils.custom_environment import cmu_humanoid_run_gaps
from utils.DMC_Gym import DMCGym
from utils.callbacks import SummaryWriterCallback
from dm_control.composer.variation import distributions
from stable_baselines3 import PPO


env_kwargs = {
    "target_velocity": 4.0,
    "gap_length": distributions.Uniform(.5, 1.25),
    "corridor_length": 100,
}

parser = argparse.ArgumentParser(description='parameters input')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--ts', type=int, default=2500000)
parser.add_argument('--bs', type=int, default=1024)
args = parser.parse_args()

env = cmu_humanoid_run_gaps()
action_spec = env.action_spec()
observation_spec = env.observation_spec()
env_wrap = DMCGym(env)

log_dir = "./logs/"

model = PPO("MlpPolicy", env_wrap,
            learning_rate=args.lr, 
            verbose=0, 
            batch_size=args.bs, 
            tensorboard_log="./logs/")


model.learn(total_timesteps=args.ts, tb_log_name="first_run", callback=SummaryWriterCallback())
model.save("ppo_humanoid")

