from backend.inference import ZombieSpawnerAgent, create_game_state, NUM_SPAWN_POINTS
import numpy as np

if __name__ == "__main__":
    agent = ZombieSpawnerAgent("zombie_spawner_model.pt")
    # Create some example test cases
    test_cases = [
        {
            "desc": "Low health (should spawn far or not at all)",
            "state": create_game_state(
                player_health=0.1,
                player_position=np.array([0, 0, 0]),
                zombies_active=8,
                spawn_distances=np.array([5, 7, 10, 20, 25]),
                time_since_spawn=20
            )
        },
        {
            "desc": "High health, low zombies (should spawn many, close & far)",
            "state": create_game_state(
                player_health=0.9,
                player_position=np.array([0, 0, 0]),
                zombies_active=3,
                spawn_distances=np.array([5, 7, 10, 20, 25]),
                time_since_spawn=2
            )
        },
        {
            "desc": "High zombies active (should spawn little or nothing)",
            "state": create_game_state(
                player_health=0.8,
                player_position=np.array([0, 0, 0]), 
                zombies_active=18,
                spawn_distances=np.array([5, 7, 10, 20, 25]),
                time_since_spawn=10
            )
        },
        {
            "desc": "Medium everything (spread out spawns)",
            "state": create_game_state(
                player_health=0.5,
                player_position=np.array([0, 0, 0]),
                zombies_active=8,
                spawn_distances=np.array([9, 11, 15, 22, 18]),
                time_since_spawn=15
            )
        }
    ]
    
    print("Testing zombie spawner agent with sampling...")
    for case in test_cases:
        # Get action using sampling (introduces some randomness)
        action = agent.select_action(case["state"])
        
        # Get distances for context
        distances = case["state"][5:5+NUM_SPAWN_POINTS]
        close_spawns = sum(action[i] for i, dist in enumerate(distances) if dist < 10)
        far_spawns = sum(action[i] for i, dist in enumerate(distances) if dist >= 20)
        
        print(f"{case['desc']} | Health: {case['state'][0]:.1f}, Zombies: {int(case['state'][4])}")
        print(f"=> Spawn Array: {action} (Total: {sum(action)}, Close: {close_spawns}, Far: {far_spawns})")
        print()
        
    print("\nTesting zombie spawner agent with deterministic actions...")
    for case in test_cases:
        # Get deterministic action (most likely)
        action = agent.get_most_likely_action(case["state"])
        
        # Get distances for context
        distances = case["state"][5:5+NUM_SPAWN_POINTS]
        close_spawns = sum(action[i] for i, dist in enumerate(distances) if dist < 10)
        far_spawns = sum(action[i] for i, dist in enumerate(distances) if dist >= 20)
        
        print(f"{case['desc']} | Health: {case['state'][0]:.1f}, Zombies: {int(case['state'][4])}")
        print(f"=> Spawn Array: {action} (Total: {sum(action)}, Close: {close_spawns}, Far: {far_spawns})")
        print()