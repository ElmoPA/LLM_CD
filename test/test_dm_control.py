from utils.DMC_Gym import DMCGym
from utils.custom_environment import cmu_humanoid_run_gaps
from dm_control import suite
import numpy as np

# Load one task:
env =  cmu_humanoid_run_gaps()
env_wrap = DMCGym(env)
print(env_wrap._action_space)

