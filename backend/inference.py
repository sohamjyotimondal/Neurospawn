import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


NUM_SPAWN_POINTS = 5
MAX_ZOMBIES_PER_POINT = 3
STATE_DIM = 1 + 3 + 1 + NUM_SPAWN_POINTS + 1  # health + pos(x,y,z) + zombies + distances + time

class RunningNormalizer:
    def __init__(self, mean, var, count):
        self.mean = mean
        self.var = var
        self.count = count
        
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

class EnhancedZombieSpawnerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.shared_net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.spawn_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, MAX_ZOMBIES_PER_POINT+1)
            ) for _ in range(NUM_SPAWN_POINTS)
        ])
        
    def forward(self, x):
        shared_features = self.shared_net(x)
        
        # We process each spawn point with its own head
        logits = []
        for head in self.spawn_heads:
            logits.append(head(shared_features))
            
        return torch.stack(logits, dim=1)

class ZombieSpawnerAgent:
    """Agent that decides when and where to spawn zombies based on game state"""
    
    def __init__(self, model_path="zombie_spawner_model.pt"):
        """Initialize the agent with a pre-trained model"""
        self.device = torch.device('cpu')
        self.policy = EnhancedZombieSpawnerPolicy().to(self.device)
        self.load_model(model_path)
        # Set model to evaluation mode
        self.policy.eval()
        
    def load_model(self, model_path):
        """Load model weights and normalizer from saved file"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
            
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.normalizer = RunningNormalizer(
                mean=checkpoint['normalizer_mean'],
                var=checkpoint['normalizer_var'],
                count=checkpoint['normalizer_count']
            )
            
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def select_action(self, state):
        """
        Select zombie spawn action based on the current game state
        
        Args:
            state: numpy array with shape (STATE_DIM,) containing:
                  [player_health, player_pos_x, player_pos_y, player_pos_z, 
                   zombies_active, spawn_point_1_dist, ..., spawn_point_5_dist, time_since_spawn]
        
        Returns:
            numpy array with shape (NUM_SPAWN_POINTS,) containing number of zombies to spawn at each point
        """
        norm_state = self.normalizer.normalize(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            logits = self.policy(state_tensor)
            
        # Sample actions from categorical distributions
        dists = [Categorical(logits=logit) for logit in logits[0]]
        
        # For inference, we can either sample or take the most likely action
        # Here we sample to maintain some variability in behavior
        actions = [dist.sample().item() for dist in dists]
        
        return np.array(actions)
    
    def get_most_likely_action(self, state):
        """
        Select the most likely (deterministic) zombie spawn action
        
        Args:
            state: numpy array with shape (STATE_DIM,)
        
        Returns:
            numpy array with shape (NUM_SPAWN_POINTS,)
        """

        norm_state = self.normalizer.normalize(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(state_tensor)
        # Take the most likely action for each spawn point
        actions = [logit.argmax().item() for logit in logits[0]]

        return np.array(actions)

def create_game_state(player_health, player_position, zombies_active, spawn_distances, time_since_spawn):
    """Helper function to create a valid game state array"""
    return np.concatenate([
        [player_health],
        player_position,
        [zombies_active],
        spawn_distances,
        [time_since_spawn]
    ])

if __name__ == "__main__":
    agent = ZombieSpawnerAgent("zombie_spawner_model.pt")
    print("Agent initialized and model loaded successfully.")