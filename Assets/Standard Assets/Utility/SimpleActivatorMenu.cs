using System;
using UnityEngine;
using UnityEngine.UI; // Add this for UI Text

namespace UnityStandardAssets.Utility
{
    public class SimpleActivatorMenu : MonoBehaviour
    {
        // Reference to a UI Text component on the Canvas
        public Text camSwitchButton;
        public GameObject[] objects;

        private int m_CurrentActiveObject;

        private void OnEnable()
        {
            // Active object starts from the first in the array
            m_CurrentActiveObject = 0;

            // Update UI text if available
            if (camSwitchButton != null && objects.Length > 0)
            {
                camSwitchButton.text = objects[m_CurrentActiveObject].name;
            }
        }

        public void NextCamera()
        {
            int nextactiveobject = m_CurrentActiveObject + 1 >= objects.Length ? 0 : m_CurrentActiveObject + 1;

            for (int i = 0; i < objects.Length; i++)
            {
                objects[i].SetActive(i == nextactiveobject);
            }

            m_CurrentActiveObject = nextactiveobject;

            // Update UI text if available
            if (camSwitchButton != null)
            {
                camSwitchButton.text = objects[m_CurrentActiveObject].name;
            }
        }
    }
}
