using System;
using System.Collections;
using System.Collections.Generic;
using System.Data.Common;
using System.IO;
using System.Linq;
//using System.Runtime.Remoting.Messaging;
using TMPro;
using UnityEngine;
using UnityEngine.UIElements;

#if ENABLE_WINMD_SUPPORT
using Windows.Storage;
#endif

public class TutorialListManager : MonoBehaviour
{
    public GameObject logger;
    private DebugOutput dbg;
    public GameObject pose_name_player;
    private string ID_before;
    public GameObject panel;
    public TextMeshPro description;
    public TextMeshPro title;
    public TextMeshPro counter;
    public PointCloudAnimation animationRenderer;

    string[] directories = null;
    
    private int idx;
    private string baseDir;
    private string objectName;
    private Matrix4x4 objectPose;

    // Start is called before the first frame update
    void Start()
    {
        #if WINDOWS_UWP
        StorageFolder o3d = KnownFolders.Objects3D;
        baseDir = o3d.Path + "\\";
        #else
        baseDir = "Assets/Resources/PointClouds/";
        #endif

        this.pose_name_player = GameObject.Find("Pose_ID_Holder");
        this.ID_before = "ID:";
        if (dbg == null)
        {
            dbg = logger.GetComponent<DebugOutput>();
        }
        panel.SetActive(true);
        // checking if it works
        //Vector3 position = Vector3.zero;
        //Quaternion orientation = Quaternion.identity;
        //Matrix4x4 Identity = Matrix4x4.TRS(position,
        //                                         orientation,
        //                                         new Vector3(1, 1, 1));
        // Debugging to see if the pannel works
        //this.SetObjectPose(Identity);
        //this.Show("QR_code_ID");
        dbg.Log("Starting tutoriallistmanager (start function called!!)");
    }

    // Update is called once per frame
    void Update()
    {
        // We continuously search for a new detected QR code
       
        this.pose_name_player = GameObject.Find("Pose_ID_Holder");

        if(this.pose_name_player.GetComponent<TextMesh>().text != this.ID_before)
        {
            dbg.Log("New Qr detected in tutorial list!!");
            // Now we set the pose and the name of the recording to the detected values
            UnityEngine.Vector3 position = this.pose_name_player.transform.position;
            UnityEngine.Quaternion rotation = this.pose_name_player.transform.rotation;
            UnityEngine.Matrix4x4 Matrixholder = UnityEngine.Matrix4x4.TRS(position, rotation, UnityEngine.Vector3.one);
            this.SetObjectPose(Matrixholder); // Contains homog. Transformation of the qr code
            this.Show(this.pose_name_player.GetComponent<TextMesh>().text);

            this.ID_before = this.pose_name_player.GetComponent<TextMesh>().text;
        }
    }

    public void UpdateInfo()
    {
        dbg.Log("Called UpdateInfo");
        if (directories == null)
        {
            dbg.Log("No object to load tutorials from");
            return;
        }
        try
        {
            counter.text = $"{idx + 1} / {directories.Length}";
            title.text = ReadTitle();
            description.text = ReadDescription();
        }
        catch (Exception e)
        {
            dbg.Log(e.Message);
        }
    }

    private string ReadDescription()
    {
        try
        {
            if (directories.Length == 0) return "";
            string path = $"{baseDir}{objectName}\\{directories[idx]}\\tutorial_info.txt";
            return string.Join("\n", System.IO.File.ReadLines(path).Skip(1));
        }
        catch(System.Exception e)
        {
            dbg.Log(e.Message);
            return e.Message;
        }
    }

    private string ReadTitle()
    {
        try
        {
            if (directories.Length == 0)
                return "Object does not have tutorials associated with it";
            string path = $"{baseDir}{objectName}\\{directories[idx]}\\tutorial_info.txt";
            dbg.Log($"Attempting to read title from path {path}");
            return System.IO.File.ReadLines(path).First().Trim();
        }
        catch(System.Exception e)
        {
            dbg.Log(e.Message);
            return e.Message;
        }
    }

    public void Show(string name)
    {
        objectName = name;
        idx = 0;
        panel.SetActive(true);
        // We want to get the tutorials in the object folder
        #if ENABLE_WINMD_SUPPORT
            StorageFolder objectFolder = KnownFolders.Objects3D;
            string dir = $"{objectFolder.Path}/{name}";
        #else
            string dir = baseDir + name;
        #endif
    
        dbg.Log($"Called show, searching {dir} for tutorials");
        try
        {
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            directories = Directory.GetDirectories(dir);
            dbg.Log("Available tutorial directories:\n");
            for (int i = 0; i < directories.Length; ++i)
            {
                directories[i] = directories[i]
                                .Substring(directories[i].LastIndexOf("\\"));
                dbg.Log($"{directories[i]}\n");
            }
        }
        catch (Exception e)
        {
            dbg.Log($"Error: {e.Message}!\n");
        }
        UpdateInfo();
    }
    public void Hide()
    {
        panel.SetActive(false);
    }

    public void Previous()
    {
        dbg.Log("Previous");
        if (idx > 0) --idx;
        UpdateInfo();
    }

    public void Next()
    {
        dbg.Log("Next");
        if (idx < directories.Length - 1) ++idx;
        UpdateInfo();
    }

    public void SetObjectPose(Matrix4x4 pose)
    {
        this.objectPose = pose;
    }

    public void TogglePlaying()
    {
        dbg.Log("Starting playback");
        animationRenderer.TogglePointCloud($"{objectName}\\{directories[idx]}", objectPose);
    }

}
