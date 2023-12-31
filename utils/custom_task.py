from dm_control.locomotion.tasks import corridors

class modifiedCorridor(corridors.RunThroughCorridor):
  def get_reward(self, physics):
    walker_xvel = physics.bind(self._walker.root_body).subtree_linvel[0]
    walker_yvel = physics.bind(self._walker.root_body).subtree_linvel[1]
    v_max = self._vel
    return min(walker_xvel, v_max) - 0.005 * (walker_xvel**2 + walker_yvel ** 2) \
      - 0.05 * walker_yvel ** 2 + 0.02