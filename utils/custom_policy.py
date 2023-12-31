import gym
import torch as th
import torch.nn as nn
from typing import Callable
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 500):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # Define shared layers with batch normalization
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space.shape[0], 5000),
            nn.ReLU(),
            nn.BatchNorm1d(5000),
            nn.Dropout(p = 0.2),
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.BatchNorm1d(2500),
            nn.Dropout(p = 0.2),
            nn.Linear(2500, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(p = 0.2),
            nn.Linear(1000, features_dim),
            nn.ReLU(),
        )
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.shared_layers(observations)

        # Return separate features for actor and critic
        # Here, using the same features for simplicity, but you can customize if needed
        return features, features

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # Process features specifically for the actor here
        # For simplicity, using the same features for actor and critic
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        # Process features specifically for the critic here
        # For simplicity, using the same features for actor and critic
        return features 

class CustomActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomFeatureExtractor(self.observation_space)

    def _build_actor(self, lr_schedule: Callable) -> None:
        super()._build_actor(lr_schedule)
        # Adding an extra layer to the actor network
        self.actor = nn.Sequential(
            *self.actor,
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU()
        )

    def _build_critic(self, lr_schedule: Callable) -> None:
        super()._build_critic(lr_schedule)
        # Adding an extra layer to the critic network
        self.critic = nn.Sequential(
            *self.critic,
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU()
        )




