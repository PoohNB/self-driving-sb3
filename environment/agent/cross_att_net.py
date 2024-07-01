import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, command_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(command_dim, hidden_dim)
        self.key = nn.Linear(latent_dim, hidden_dim)
        self.value = nn.Linear(latent_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent_vector, command_embedding):
        query = self.query(command_embedding).unsqueeze(0)
        key = self.key(latent_vector).unsqueeze(0)
        value = self.value(latent_vector).unsqueeze(0)
        attn_output, _ = self.attention(query, key, value)
        attn_output = attn_output.squeeze(0)
        return self.fc(attn_output)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, latent_dim, command_dim, hidden_dim):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=latent_dim)
        self.latent_dim = latent_dim
        self.cross_attention = CrossAttention(latent_dim, command_dim, hidden_dim)

    def forward(self, observations):
        # Split the observation into latent vector and command
        latent_vector = observations[:, :self.latent_dim]
        command_embedding = observations[:, self.latent_dim:]
        # Apply cross-attention
        attn_output = self.cross_attention(latent_vector, command_embedding)
        return attn_output

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs,
                                           features_extractor_class=CustomFeatureExtractor,
                                           features_extractor_kwargs=dict(
                                               latent_dim=128,  # Your latent space dimension
                                               command_dim=1,  # Your command embedding dimension
                                               hidden_dim=64  # Hidden dimension for cross-attention
                                           ))

