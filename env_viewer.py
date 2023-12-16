from env_lib import cmu_humanoid_run_gaps
from dm_control import suite
from dm_control import viewer
import numpy as np

env = cmu_humanoid_run_gaps()
action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

# Launch the viewer application.
viewer.launch(env, policy=random_policy)
print("Launched!")