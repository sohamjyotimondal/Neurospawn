// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine;
// using UnityEngine.SceneManagement;
// using UnityEngine.UI;
// using UnityStandardAssets.Characters.FirstPerson;

// public class Player : MonoBehaviour {
// 	HealthManager healthManager;
// 	bool isDestroyed = false;
// 	public GameObject deadScreen;

// 	void Start() {
// 		healthManager = GetComponent<HealthManager>();
// 		deadScreen = GameObject.Find("UI/InGameUI/DeadScreen");
// 	}

// 	void Update() {
// 		if(healthManager.IsDead && !isDestroyed) {
// 			isDestroyed = true;

// 			StartCoroutine(ShowDeadScreen());

// 			MonoBehaviour[] scripts = GetComponentsInChildren<MonoBehaviour>();

// 			foreach(MonoBehaviour script in scripts) {
// 				// Disable all weapons
// 				if(script is WeaponBase) {
// 					DisableWeapon((WeaponBase)script);
// 				}
// 				// Deactivate player controls
// 				else if(script is FirstPersonController) {
// 					DisableController((FirstPersonController)script);
// 				}
// 			}

// 			Invoke("GotoDieScene", 2.0f);

// 		}
// 	}

// 	private void GotoDieScene(){
// 		SceneManager.LoadScene(3);
// 	}

// 	void DisableWeapon(WeaponBase weapon) {
// 		weapon.IsEnabled = false;
// 	}

// 	void DisableController(FirstPersonController controller) {
// 		controller.enabled = false;
// 	}

// 	IEnumerator ShowDeadScreen() {
// 		deadScreen.SetActive(true);

// 		Image image = deadScreen.GetComponent<Image>();
// 		Color origColor = image.color;

// 		for(float alpha = 0.0f; alpha <= 1.1f; alpha += 0.1f) {
// 			image.color = new Color(origColor.r, origColor.g, origColor.b, alpha);
// 			yield return new WaitForSeconds(0.1f);
// 		}

// 		yield break;
// 	}

// 	void OnControllerColliderHit(ControllerColliderHit hit) {
// 		if(hit.gameObject.tag == "BulletCase") {
// 			Physics.IgnoreCollision(GetComponent<Collider>(), hit.gameObject.GetComponent<Collider>());
// 		}
// 	}
// }






using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.Networking;
using UnityStandardAssets.Characters.FirstPerson;
using System;

public class Player : MonoBehaviour {
    HealthManager healthManager;
    bool isDestroyed = false;
    public GameObject deadScreen;

    // Variables from server
    public float playerHealth = 100.0f;
    public Vector3 playerPosition = Vector3.zero;
    public int zombiesActive = 5;
    public float[] spawnDistances = new float[] {5f, 7f, 10f, 20f, 25f};
    public int timeSinceSpawn = 10;
    public bool deterministic = false;
    
    // Server configuration
    [SerializeField] private string serverUrl = "http://localhost:5000/predict";
    [SerializeField] private float updateInterval = 10.0f;
    private float lastUpdateTime = 0f;

    void Start() {
        healthManager = GetComponent<HealthManager>();
        deadScreen = GameObject.Find("UI/InGameUI/DeadScreen");
        
        // Log initial values
        Debug.Log($"[Player] Start - Health: {playerHealth}, Position: {playerPosition}, Zombies Active: {zombiesActive}");
        Debug.Log($"[Player] Spawn Distances: {string.Join(", ", spawnDistances)}");
        Debug.Log($"[Player] Time Since Last Spawn: {timeSinceSpawn}, Deterministic: {deterministic}");
    }

    void Update() {
        // Update current position
        playerPosition = transform.position;
        
        // Update playerHealth from healthManager if available
        if(healthManager != null) {
            playerHealth = healthManager.Health; // Fixed from CurrentHealth to Health
        }
        
        // Periodically request spawn data from server
        if (Time.time - lastUpdateTime > updateInterval) {
            lastUpdateTime = Time.time;
            StartCoroutine(FetchServerData());
        }
        
        if(healthManager.IsDead && !isDestroyed) {
            isDestroyed = true;
            
            Debug.Log("[Player] Player is dead. Showing dead screen and disabling controls.");
            
            StartCoroutine(ShowDeadScreen());

            MonoBehaviour[] scripts = GetComponentsInChildren<MonoBehaviour>();

            foreach(MonoBehaviour script in scripts) {
                // Disable all weapons
                if(script is WeaponBase) {
                    DisableWeapon((WeaponBase)script);
                    Debug.Log($"[Player] Disabled weapon: {script.GetType().Name}");
                }
                // Deactivate player controls
                else if(script is FirstPersonController) {
                    DisableController((FirstPersonController)script);
                    Debug.Log($"[Player] Disabled controller: {script.GetType().Name}");
                }
            }

            Invoke("GotoDieScene", 2.0f);
        }
    }

    private void GotoDieScene(){
        Debug.Log("[Player] Loading death scene (Scene 3)");
        SceneManager.LoadScene(3);
    }

    void DisableWeapon(WeaponBase weapon) {
        weapon.IsEnabled = false;
    }

    void DisableController(FirstPersonController controller) {
        controller.enabled = false;
    }

    IEnumerator ShowDeadScreen() {
        deadScreen.SetActive(true);

        Image image = deadScreen.GetComponent<Image>();
        Color origColor = image.color;

        for(float alpha = 0.0f; alpha <= 1.1f; alpha += 0.1f) {
            image.color = new Color(origColor.r, origColor.g, origColor.b, alpha);
            yield return new WaitForSeconds(0.1f);
        }

        yield break;
    }

    void OnControllerColliderHit(ControllerColliderHit hit) {
        if(hit.gameObject.tag == "BulletCase") {
            Physics.IgnoreCollision(GetComponent<Collider>(), hit.gameObject.GetComponent<Collider>());
        }
    }

    // Method to update variables from server data
    public void UpdateFromServer(float health, Vector3 position, int zombies, float[] distances, int timeSpawn, bool isDeterministic) {
        playerHealth = health;
        playerPosition = position;
        zombiesActive = zombies;
        spawnDistances = distances;
        timeSinceSpawn = timeSpawn;
        deterministic = isDeterministic;

        Debug.Log($"[Player] Updated from server - Health: {playerHealth}, Position: {playerPosition.x:F2},{playerPosition.y:F2},{playerPosition.z:F2}");
        Debug.Log($"[Player] Zombies Active: {zombiesActive}, Time Since Spawn: {timeSinceSpawn}");
        Debug.Log($"[Player] Spawn Distances: {string.Join(", ", Array.ConvertAll(spawnDistances, d => d.ToString("F1")))}");
    }
    
    // Fetch data from the server
    IEnumerator FetchServerData() {
        // Create the request data
        RequestData requestData = new RequestData {
            player_health = playerHealth,
            player_position = new float[] { playerPosition.x, playerPosition.y, playerPosition.z },
            zombies_active = zombiesActive,
            spawn_distances = spawnDistances,
            time_since_spawn = timeSinceSpawn,
            deterministic = deterministic
        };
        
        // Convert to JSON
        string jsonData = JsonUtility.ToJson(requestData);
        Debug.Log($"[Player] Sending request to server: {jsonData}");
        
        // Create the web request
        using (UnityWebRequest request = new UnityWebRequest(serverUrl, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            
            // Send the request
            yield return request.SendWebRequest();
            
            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[Player] Server request error: {request.error}");
            }
            else
            {
                Debug.Log($"[Player] Server response: {request.downloadHandler.text}");
                
                try {
                    // Parse the response
                    ServerResponse response = JsonUtility.FromJson<ServerResponse>(request.downloadHandler.text);
                    
                    if (response.success)
                    {
                        Debug.Log($"[Player] Spawn counts received: {string.Join(", ", response.spawn_counts)}");
                        Debug.Log($"[Player] Total spawns: {response.total_spawns}, Close: {response.close_spawns}, Far: {response.far_spawns}");
                        
                        // Here you would trigger zombie spawning based on the response
                        // For example: SpawnZombies(response.spawn_counts);
                    }
                    else
                    {
                        Debug.LogError($"[Player] Server returned error: {response.error}");
                    }
                }
                catch (Exception e) {
                    Debug.LogError($"[Player] Error parsing server response: {e.Message}");
                }
            }
        }
    }
    
    // Helper classes for JSON serialization
    [Serializable]
    private class RequestData {
        public float player_health;
        public float[] player_position;
        public int zombies_active;
        public float[] spawn_distances;
        public int time_since_spawn;
        public bool deterministic;
    }
    
    [Serializable]
    private class ServerResponse {
        public int[] spawn_counts;
        public int total_spawns;
        public int close_spawns;
        public int far_spawns;
        public bool success;
        public string error;
    }
}
