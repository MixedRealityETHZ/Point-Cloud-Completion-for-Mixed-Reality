using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


/// <summary>
/// Handles arbitrary object behaviour spawned by the object manager panel
/// </summary>
public class ObjectManager : MonoBehaviour
{
    public GameObject objectManagerPanel;
    public GameObject logger;
    private DebugOutput dbg;

    [SerializeField]
    private GameObject realObjects;

    private Dictionary<string, GameObject> originalObjects = new Dictionary<string, GameObject>();
    private Dictionary<string, GameObject> spawnedObjects = new Dictionary<string, GameObject>();

    private GameObject lastInteractedObject;

    private GameObject userDefinedOjbect;

    /// <summary>
    /// Gives each instance a new ID
    /// </summary>
    private int n_clones;

    // Start is called before the first frame update
    void Start()
    {
        if (dbg == null)
        {
            dbg = logger.GetComponent<DebugOutput>();
        }
        Debug.Log("From objectmanager");
        foreach (Transform child in realObjects.transform)
        {
            child.gameObject.SetActive(false);
            originalObjects.Add(child.name, child.gameObject);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    /// <summary>
    /// Spawns an instance of the assigned object
    /// </summary>
    public void SpawnObject(string type)
    {
        dbg.Log($"Called SpawnObject with type '{type}'");
        if (!originalObjects.TryGetValue(type, out GameObject objectModel))
        {
            Debug.Log($"Object type not available: {type}");
            return;
        }
        GameObject clone = Instantiate(
            objectModel,
            objectManagerPanel.transform.position + new Vector3(.0f, .1f, .0f),
            objectManagerPanel.transform.rotation * objectModel.transform.rotation,
            objectModel.transform.parent);
        clone.name = $"{objectModel.name}-{n_clones++}";
        clone.transform.localScale *= 2;
        spawnedObjects.Add(clone.name, clone);
        clone.SetActive(true);
    }

    /// <summary>
    /// Sets the tracked last interacted object to the last object hovered over (normally triggered in object manipulator)
    /// </summary>
    /// <param name="obj"></param>
    public void SetLastInteractedObject(GameObject obj)
    {
        lastInteractedObject = obj;
    }

    /// <summary>
    /// Removes the last interacted object from the scene
    /// </summary>
    public void RemoveLastInteractedObject()
    {
        spawnedObjects.Remove(lastInteractedObject.name);
        Destroy(lastInteractedObject);
        lastInteractedObject = null;
    }

    /// <summary>
    /// Gives the name of the last interacted object
    /// </summary>
    /// <returns></returns>
    public string GetLastInteractedObjectName()
    {
        if (lastInteractedObject == null)
        {
            return "None";
        }
        return lastInteractedObject.name;
    }

    /// <summary>
    /// Get original objects by name
    /// </summary>
    /// <returns></returns>
    public GameObject GetOriginalObject(string name)
    {
        if (originalObjects.TryGetValue(name, out GameObject obj))
        {
            return obj;
        }
        return null;
    }

    /// <summary>
    /// Get spawned objects by name
    /// </summary>
    /// <returns></returns>
    public GameObject GetSpawnedObject(string name)
    {
        if (spawnedObjects.TryGetValue(name, out GameObject obj))
        {
            return obj;
        }
        return null;
    }

    /// <summary>
    /// Get a list of spawned objects
    /// </summary>
    /// <returns></returns>
    public List<GameObject> GetSpawnedObjects()
    {
        return spawnedObjects.Values.ToList();
    }

    /// <summary>
    /// Only use for when the user is defining an object with the bounding box
    /// </summary>
    public void AddSpawnedObject(GameObject userObj)
    {
        if (spawnedObjects.ContainsKey("userdefined"))
        {
            // Remove old user created object if there is one
            spawnedObjects.Remove("userdefined");
        }
        userObj.name = "userdefined";
        spawnedObjects.Add(userObj.name, userObj);
        Debug.Log("Added object to the list of spawned objects");

        userDefinedOjbect = userObj;
    }

    public GameObject GetUserDefinedObject()
    {
        return userDefinedOjbect;
    }

    public void ActivateRealObjects()
    {
        realObjects.SetActive(true);
    }

    public void DeactivateRealObjects()
    {
        realObjects.SetActive(false);
    }

}
