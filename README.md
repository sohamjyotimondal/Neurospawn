
#REVENANT - NEUROSPAWN
## Reinforcement Learning Approach

This zombie spawner uses a deep reinforcement learning agent to decide how many zombies to spawn at each of five spawn points, based on the current game state. The agent observes a normalized state vector containing player health, position, active zombie count, distances to each spawn point, and time since last spawn. A custom reward function guides the agent to balance challenge and fairness: it penalizes overwhelming the player, encourages spawning at distant points when health is low, and rewards diverse, strategic spawning when the player is strong. The agent is trained in a simulated environment using policy gradient methods with entropy regularization to ensure both effective learning and varied, adaptive gameplay behavior.

## Neural Network Architecture

The policy network consists of a shared feature extractor followed by separate heads for each spawn point:

- **Input:** Normalized state vector of size 11 (player health, position (x, y, z), zombies active, 5 spawn point distances, time since last spawn).
- **Shared Layers:**  
  - Linear (11 → 256), LayerNorm, ReLU  
  - Linear (256 → 256), LayerNorm, ReLU  
  - Linear (256 → 128), ReLU  
- **Spawn Point Heads:** For each of the 5 spawn points, a head:
  - Linear (128 → 64), ReLU  
  - Linear (64 → 4) — outputs logits for spawning 0, 1, 2, or 3 zombies  
- **Output:** Tensor of shape (batch_size, 5, 4), representing action logits for each spawn point.

This architecture enables the agent to learn both global game context and specialized spawning strategies for each location, resulting in adaptive and context-aware zombie spawning.
