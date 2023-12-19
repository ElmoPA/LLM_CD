from utils.custom_environment import cmu_humanoid_run_gaps
from dm_control import viewer
from dm_control.composer.variation import distributions
import numpy as np

env = cmu_humanoid_run_gaps(gap_length=distributions.Uniform(0, 2.0))
action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

# Launch the viewer application.
viewer.launch(env, policy=random_policy)
