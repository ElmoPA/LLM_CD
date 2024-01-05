import enum
import warnings
import weakref

from absl import logging
from dm_control import mjcf
from dm_control.composer import observation
from dm_control.rl import control
import dm_env
import numpy as np
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

  return CustomEnvironment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)

class CustomEnvironment(composer.Environment):
  def step(self, action):
    """Updates the environment using the action and returns a `TimeStep`."""
    if self._reset_next_step:
      self._reset_next_step = False
      return self.reset()

    self._hooks.before_step(self._physics_proxy, action, self._random_state)
    self._observation_updater.prepare_for_next_control_step()

    try:
      for i in range(self._n_sub_steps):
        self._substep(action)
        # The final observation update must happen after all the hooks in
        # `self._hooks.after_step` is called. Otherwise, if any of these hooks
        # modify the physics state then we might capture an observation that is
        # inconsistent with the final physics state.
        if i < self._n_sub_steps - 1:
          self._observation_updater.update()
      physics_is_divergent = False
    except control.PhysicsError as e:
      if not self._raise_exception_on_physics_error:
        logging.warning(e)
        physics_is_divergent = True
      else:
        raise

    self._hooks.after_step(self._physics_proxy, self._random_state)
    self._observation_updater.update()

    if not physics_is_divergent:
      reward = self._task.get_reward(self._physics_proxy, action)
      discount = self._task.get_discount(self._physics_proxy)
      terminating = (
          self._task.should_terminate_episode(self._physics_proxy)
          or self._physics.time() >= self._time_limit
      )
    else:
      reward = 0.0
      discount = 0.0
      terminating = True

    obs = self._observation_updater.get_observation()

    if not terminating:
      return dm_env.TimeStep(dm_env.StepType.MID, reward, discount, obs)
    else:
      self._reset_next_step = True
      return dm_env.TimeStep(dm_env.StepType.LAST, reward, discount, obs)
