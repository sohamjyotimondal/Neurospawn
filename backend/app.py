from flask import Flask, request, jsonify
import numpy as np
import os
import logging
from backend.inference import ZombieSpawnerAgent, create_game_state


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'zombie_spawner_model.pt')
agent = None

def get_agent():
    global agent
    if agent is None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            agent = ZombieSpawnerAgent(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    return agent

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Zombie Spawner API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                h1 { color: #333; }
                pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; }
                .endpoint { margin: 20px 0; padding: 10px; border-left: 4px solid #4CAF50; }
            </style>
        </head>
        <body>
            <h1>Zombie Spawner API</h1>
            <div class="endpoint">
                <h2>POST /predict</h2>
                <p>Get zombie spawn decisions based on game state</p>
                <h3>Request Format:</h3>
                <pre>
{
    "player_health": 0.8,
    "player_position": [0, 0, 0],
    "zombies_active": 18,
    "spawn_distances": [5, 7, 10, 20, 25],
    "time_since_spawn": 10,
    "deterministic": false
}
                </pre>
                <h3>Response Format:</h3>
                <pre>
{
    "spawn_counts": [0, 0, 1, 0, 0],
    "total_spawns": 1,
    "close_spawns": 0,
    "far_spawns": 0,
    "success": true
}
                </pre>
            </div>
        </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "success": False}), 400
  
        try:
            player_health = float(data.get('player_health', 0.8))
            player_position = np.array(data.get('player_position', [0, 0, 0]), dtype=np.float32)
            zombies_active = int(data.get('zombies_active', 0))
            spawn_distances = np.array(data.get('spawn_distances', [15, 15, 15, 15, 15]), dtype=np.float32)
            time_since_spawn = int(data.get('time_since_spawn', 0))
            deterministic = bool(data.get('deterministic', False))
            
            if len(player_position) != 3:
                return jsonify({"error": "player_position must have 3 elements", "success": False}), 400
            if len(spawn_distances) != 5:
                return jsonify({"error": "spawn_distances must have 5 elements", "success": False}), 400
                
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid input format: {str(e)}", "success": False}), 400
        
        state = create_game_state(
            player_health=player_health,
            player_position=player_position,
            zombies_active=zombies_active,
            spawn_distances=spawn_distances,
            time_since_spawn=time_since_spawn
        )
        
        try:
            agent = get_agent()
        except Exception as e:
            return jsonify({"error": f"Model initialization error: {str(e)}", "success": False}), 500
    
        if deterministic:
            spawn_counts = agent.get_most_likely_action(state).tolist()
        else:
            spawn_counts = agent.select_action(state).tolist()
            
        total_spawns = sum(spawn_counts)
        close_spawns = sum(spawn_counts[i] for i, dist in enumerate(spawn_distances) if dist < 10)
        far_spawns = sum(spawn_counts[i] for i, dist in enumerate(spawn_distances) if dist >= 20)
 
        return jsonify({
            "spawn_counts": spawn_counts,
            "total_spawns": total_spawns,
            "close_spawns": close_spawns,
            "far_spawns": far_spawns,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}", "success": False}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:

        get_agent()
        return jsonify({"status": "healthy", "success": True})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e), "success": False}), 500

if __name__ == "__main__":

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)