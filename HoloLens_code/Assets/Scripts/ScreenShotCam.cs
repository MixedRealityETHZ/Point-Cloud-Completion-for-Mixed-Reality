using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if WINDOWS_UWP
using Windows.Storage;
#endif

public class ScreenShotCam : MonoBehaviour
{
    private int counter = 0;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void SaveCameraView(Camera cam, int idx)
    {
        RenderTexture screenTexture = new RenderTexture(Screen.width, Screen.height, 16);
        var oldTexture = cam.targetTexture;
        cam.targetTexture = screenTexture;
        RenderTexture.active = screenTexture;
        cam.Render();
        Texture2D renderedTexture = new Texture2D(Screen.width, Screen.height);
        renderedTexture.ReadPixels(new Rect(0, 0, Screen.width, Screen.height), 0, 0);
        RenderTexture.active = null;
        byte[] byteArray = renderedTexture.EncodeToPNG();

        cam.targetTexture = oldTexture;
        string dir = "";
#if WINDOWS_UWP
            Windows.Storage.StorageFolder storageFolder = KnownFolders.Objects3D;
            dir = storageFolder.Path + "\\";
#endif
        try
        {
            System.IO.File.WriteAllBytes($"{dir}screenshot_{idx}_{counter++}.png", byteArray);
        }
        catch(Exception e)
        {
            Debug.Log($"Couldn't save screenshot: {e.Message}");
        }
    }
}
