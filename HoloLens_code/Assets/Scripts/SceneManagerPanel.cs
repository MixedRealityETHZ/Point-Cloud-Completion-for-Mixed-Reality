using UnityEngine;

/// <summary>
/// Handles functionalities provided by a scene manager panel
/// </summary>
public class SceneManagerPanel : MonoBehaviour
{
    [SerializeField]
    private GameObject panel;

    private void Start()
    {
        panel.SetActive(false);
    }

    /// <summary>
    /// Toggles a manager panel to control specific aspects of the scene
    /// </summary>
    public void ToggleSceneManager()
    {
        panel.SetActive(!panel.activeSelf);
    }
}
