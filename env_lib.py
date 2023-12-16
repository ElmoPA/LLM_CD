import functools

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import cmu_humanoid

def cmu_humanoid_run_gaps(random_state=None, **kwargs):
  """Requires a CMU humanoid to run down a corridor with gaps."""
  task = {
    "target_velocity": 3.0,
    "gap_length": (.5, 1.25),
    "corridor_length": 100,
  }
  for param, value in kwargs:
    task[param] = value
  gap_length = task["gap_length"]
  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
  # platforms are uniformly randomized.
  arena = corr_arenas.GapsCorridor(
      platform_length=distributions.Uniform(.3, 2.5),
      gap_length=distributions.Uniform(gap_length[0], gap_length[1]),
      corridor_width=10,
      corridor_length=task["corridor_length"])

  # Build a task that rewards the agent for running down the corridor at a
  # specific velocity.
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(0.5, 0, 0),
      target_velocity=task["target_velocity"],
      physics_timestep=0.005,
      control_timestep=0.03)

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)
