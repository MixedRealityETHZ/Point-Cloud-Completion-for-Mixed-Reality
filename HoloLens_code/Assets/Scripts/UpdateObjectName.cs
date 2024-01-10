using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class UpdateObjectName : MonoBehaviour
{
    [SerializeField]
    private TextMeshPro objectLabelText;
    [SerializeField]
    private ObjectManager objectManager;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        objectLabelText.text = objectManager.GetLastInteractedObjectName();
    }
}
