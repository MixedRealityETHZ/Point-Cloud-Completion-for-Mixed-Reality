using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class DebugOutput : MonoBehaviour
{
    public TextMeshPro output;
    private int outputLimit = 20;
    private Queue<(DateTime, string)> logs;
    private bool changed = false;
    // Start is called before the first frame update
    void Start()
    {
    }

    DebugOutput()
    {
        logs = new Queue<(DateTime, string)>();
    }

    // Update is called once per frame
    void Update()
    {
        if (!changed) return;
        output.text = "";
        foreach ((DateTime timestamp, string msg) in logs)
        {
            output.text += string.Format("[{0}] ", timestamp.ToString("HH:mm:ss"));
            output.text += msg + "\n";
            output.text += new string('-', 100) + "\n";
        }
        changed = false;
    }

    public void Log(string msg)
    {
        if (logs.Count > outputLimit)
        {
            logs.Dequeue();
        }
        logs.Enqueue((DateTime.Now, msg));
        changed = true;
    }
}
