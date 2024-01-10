using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PointOfReferenceHandle : MonoBehaviour
{
    [SerializeField]
    private GameObject handle;

    // Start is called before the first frame update
    void Start()
    {
        handle.SetActive(false);
    }

    public void ToggleHandle()
    {
        handle.SetActive(!handle.activeSelf);
    }
}
