using UnityEngine;
using UnityEngine.SceneManagement;
using UnityStandardAssets.CrossPlatformInput;

public class ForcedReset : MonoBehaviour
{
    void Update()
    {
        // Check if the "ResetObject" button is pressed
        if (CrossPlatformInputManager.GetButtonDown("ResetObject"))
        {
            // Reload the currently active scene
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }
    }
}
