import functools
from .custom_task import modifiedCorridor
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import cmu_humanoid

def cmu_humanoid_run_gaps(random_state=None, **kwargs):
  """Requires a CMU humanoid to run down a corridor with gaps."""
  task = {
    "target_velocity": 3.0,
    "corridor_length": 100,
    "gap_length": distributions.Uniform(.3, 2.5,)
  }
  for param, value in kwargs.items():
    task[param] = value
  gap_length = task["gap_length"]
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=False)})

  arena = corr_arenas.GapsCorridor(
      platform_length=distributions.Uniform(.5, 2.5),
      gap_length=gap_length,
      corridor_width=10,
      corridor_length=task["corridor_length"])

  task = modifiedCorridor(
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




