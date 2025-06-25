import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
NUM_SPAWN_POINTS = 5
MAX_ZOMBIES_PER_POINT = 3
STATE_DIM = 1 + 3 + 1 + NUM_SPAWN_POINTS + 1  # health + pos(x,y,z) + zombies + distances + time

# Initial normalization values
STATE_MEAN = np.array([0.5, 0, 0, 0, 10, 15, 15, 15, 15, 15, 30])
STATE_STD = np.array([0.3, 30, 30, 30, 8, 8, 8, 8, 8, 8, 20])

class RunningNormalizer:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = eps
        
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        delta = batch_mean - self.mean
        total = self.count + x.shape[0]
        
        self.mean += delta * x.shape[0] / total
        m_a = self.var * self.count
        m_b = batch_var * x.shape[0]
        M2 = m_a + m_b + delta**2 * self.count * x.shape[0] / total
        self.var = M2 / total
        self.count = total
        
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

def normalize_state(state):
    return (state - STATE_MEAN) / (STATE_STD + 1e-8)

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
        logits = []
        for head in self.spawn_heads:
            logits.append(head(shared_features))
            
        return torch.stack(logits, dim=1)

class ZombieEnvSimulator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.player_health = 1.0
        self.zombies_active = 0
        self.spawn_points = np.random.uniform(5, 25, NUM_SPAWN_POINTS)
        self.time_since_spawn = 0
        return self.get_state()
        
    def step(self, action): #simulation of the game environment stwp by step
        self.zombies_active = min(20, self.zombies_active + sum(action))
        damage = 0.01 * sum(action) + 0.005 * self.zombies_active
        self.player_health = max(0.1, self.player_health - damage)
        
        # let zombies die
        self.zombies_active = max(0, self.zombies_active - np.random.randint(0, 3))
        
        # zombie movement
        self.spawn_points += np.random.uniform(-2, 2, NUM_SPAWN_POINTS)
        self.spawn_points = np.clip(self.spawn_points, 3, 30)
        
        # Update time
        self.time_since_spawn = 0 if sum(action) > 0 else self.time_since_spawn + 1
        
        # Health recovery when few zombies
        if self.zombies_active < 5:
            self.player_health = min(1.0, self.player_health + 0.01)
            
        return self.get_state()
    
    def get_state(self):
        return np.concatenate([
            [self.player_health],
            np.random.uniform(-30, 30, 3),  # Simulated position
            [self.zombies_active],
            self.spawn_points,
            [self.time_since_spawn]
        ])

def select_action_with_entropy(policy, state, eval_mode=False):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    if eval_mode:
        with torch.no_grad():
            logits = policy(state_tensor)
    else:
        logits = policy(state_tensor)
    
    dists = [Categorical(logits=logit) for logit in logits[0]]
    actions = [dist.sample() for dist in dists]
    
    if not eval_mode:
        log_probs = torch.stack([dist.log_prob(action) for dist, action in zip(dists, actions)])
        entropies = torch.stack([dist.entropy() for dist in dists])
        log_prob_sum = log_probs.sum()
        entropy_sum = entropies.sum()
    else:
        log_prob_sum = None
        entropy_sum = None
    
    return np.array([a.item() for a in actions]), log_prob_sum, entropy_sum

def strategic_reward(state, action):
    player_health = state[0]
    zombies_active = state[4]
    spawn_distances = state[5:5+NUM_SPAWN_POINTS]
    
    reward = 0
    total_spawned = sum(action)
    
    # We categorize points based on distance (close and far)
    close_spawns = sum(action[i] for i, dist in enumerate(spawn_distances) if dist < 10)
    medium_spawns = sum(action[i] for i, dist in enumerate(spawn_distances) if 10 <= dist < 20)
    far_spawns = sum(action[i] for i, dist in enumerate(spawn_distances) if dist >= 20)
    
    # 1) If there are too many zombies, don't spawn many zombies
    if zombies_active > 12:
        if total_spawned == 0:
            reward += 15  # Strong reward for not spawning when too many zombies
        elif total_spawned == 1:
            reward += 5   # Small reward for minimal spawning when too many zombies
        else:
            reward -= 8 * total_spawned  # Strong penalty scales with number spawned
    
    # 2) If player health is low, zombies spawn on further spawn points
    if player_health < 0.4:
        # Penalize close spawns when health is low
        reward -= 10 * close_spawns
        
        # Reward far spawns when health is low
        reward += 4 * far_spawns
        
        # Extra reward for not spawning at all when health is very low
        if player_health < 0.2 and total_spawned == 0:
            reward += 20
    
    # 3) If player health is high and zombies active is low, spawn higher numbers
    if player_health > 0.7 and zombies_active < 6:
        # Reward for good distribution (both close and far)
        if close_spawns > 0 and far_spawns > 0:
            reward += 20  # Very strong reward for having both close and far spawns
        
        # Encourage higher spawn counts in ideal conditions
        if total_spawned > 6:
            reward += 15
        elif total_spawned < 3:
            reward -= 10  # Penalize not spawning enough in ideal conditions
    
    # 4) If zombies active is high, minimal or no spawning
    if zombies_active > 15:
        if total_spawned == 0:
            reward += 25  # Very strong reward for not spawning at all
        elif total_spawned == 1 and far_spawns == 1:
            reward += 10  # Good reward for one far spawn
        else:
            reward -= 6 * total_spawned  # Strong penalty for too many spawns
    
    # 5) If player health is high, zombies active is low, spawn to close and far points
    if player_health > 0.7 and zombies_active < 5:
        # Must have some close spawns
        if close_spawns == 0:
            reward -= 10
        
        # Must have some far spawns
        if far_spawns == 0:
            reward -= 10
            
        # Reward for diversity in spawn distances
        active_points = sum(1 for a in action if a > 0)
        if active_points >= 3:
            reward += 12  # Reward for using multiple spawn points
    
    # General penalties for excessive spawning
    if total_spawned > 10:
        reward -= (total_spawned - 10) * 5
        
    return reward

def train_rl(policy, epochs=700, batch_size=128, lr=3e-4):
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    env = ZombieEnvSimulator()
    normalizer = RunningNormalizer(STATE_DIM)
    
    # plotting
    avg_rewards = []
    
    # Initialize normalizer with dataset
    init_states = [env.reset() for _ in range(1000)]
    for state in init_states:
        normalizer.update(state[None,:])
    
    for epoch in range(epochs):
        states, actions, rewards, log_probs, entropies = [], [], [], [], []
        
 
        for _ in range(batch_size):
            state = env.reset() if np.random.random() < 0.3 else env.get_state()
            normalizer.update(state[None,:])
            norm_state = normalizer.normalize(state)
  
            action, log_prob, entropy = select_action_with_entropy(policy, norm_state)
            next_state = env.step(action)
            reward = strategic_reward(state, action)
            
            # Store transitions
            states.append(norm_state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            
        # Calculate advantages
        rewards = np.array(rewards)
        advantages = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        advantages = torch.FloatTensor(advantages).to(device)  
        # Policy update with entropy regularization
        optimizer.zero_grad()
        
        # Stack all tensors for vectorized operations
        policy_loss = torch.stack(log_probs) * (-advantages)
        entropy_loss = -torch.stack(entropies)
        
        # Add entropy regularization to encourage exploration
        entropy_coef = max(0.01, 0.3 * (1 - epoch/epochs))  # High at start, decay over time
        total_loss = policy_loss.mean() + entropy_coef * entropy_loss.mean()
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Store average reward for plotting
        avg_rewards.append(np.mean(rewards))
    
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Avg Reward: {np.mean(rewards):.2f}, Entropy Coef: {entropy_coef:.3f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.savefig('zombie_rl_training.png')
    plt.close()
    
    return normalizer

def test_agent(policy, normalizer=None):
    test_cases = [
        {
            "desc": "Low health (should spawn far or not at all)",
            "state": np.concatenate([[0.1], [0,0,0], [8], [5, 7, 10, 20, 25], [20]])
        },
        {
            "desc": "High health, low zombies (should spawn many, close & far)",
            "state": np.concatenate([[0.9], [0,0,0], [3], [5, 7, 10, 20, 25], [2]])
        },
        {
            "desc": "High zombies active (should spawn little or nothing)",
            "state": np.concatenate([[0.8], [0,0,0], [18], [5, 7, 10, 20, 25], [10]])
        },
        {
            "desc": "Medium everything (spread out spawns)",
            "state": np.concatenate([[0.5], [0,0,0], [8], [9, 11, 15, 22, 18], [15]])
        }
    ]
    
    for case in test_cases:
        if normalizer:
            state = normalizer.normalize(case["state"])
        else:
            state = normalize_state(case["state"])
            
        action, _, _ = select_action_with_entropy(policy, state, eval_mode=True)
        reward = strategic_reward(case["state"], action)
        
        # Get distances for context
        distances = case["state"][5:10]
        close_spawns = sum(action[i] for i, dist in enumerate(distances) if dist < 10)
        far_spawns = sum(action[i] for i, dist in enumerate(distances) if dist >= 20)
        
        print(f"{case['desc']} | Health: {case['state'][0]:.1f}, Zombies: {int(case['state'][4])}")
        print(f"=> Spawn Array: {action} (Total: {sum(action)}, Close: {close_spawns}, Far: {far_spawns})")
        print(f"=> Expected Reward: {reward:.2f}\n")

def save_model(policy, normalizer, path):
    torch.save({
        'model_state_dict': policy.state_dict(),
        'normalizer_mean': normalizer.mean,
        'normalizer_var': normalizer.var,
        'normalizer_count': normalizer.count
    }, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    print("Using device:", device)
    print("Initializing agent...")
    policy = EnhancedZombieSpawnerPolicy().to(device)
    
    print("Training agent...")
    normalizer = train_rl(policy, epochs=700, batch_size=128, lr=3e-4)
    
    print("Testing agent...")
    test_agent(policy, normalizer)
    
    save_model(policy, normalizer, "zombie_spawner_model.pt")