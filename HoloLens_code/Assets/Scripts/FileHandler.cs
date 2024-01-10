using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Threading.Tasks;
using System.Xml.Serialization;
using UnityEngine.Events;
//using System.Runtime.Remoting.Messaging;
using Microsoft.MixedReality.Toolkit.Utilities;
using System.CodeDom;

namespace Tutorials
{

    /// <summary>
    /// Static class that provides logic for handling files and the current state of the loaded animations in the editor through its field AnimationListInstance, which exists only once in this context and can be accessed by any other class in this namespace.
    /// </summary>
    public static class FileHandler
    {
        private static string RECORDINGS_DIRECTORY = "Recordings";
        private static string ANIMATIONFILE_PREFIX = "HandAnimation";
        private static string DATAFILE_NAME = "datafile.xml";

        /// <summary>
        /// Creates the directory for the recorded data in the persistent data path, if it doesn't already exist.
        /// </summary>
        public static void CreateDirectory()
        {
            string path = Path.Combine(Application.persistentDataPath, RECORDINGS_DIRECTORY);

            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
        }

        /// <summary>
        /// Returns the file path of a file in the recording directory when its name is passed as a parameter
        /// </summary>
        /// <param name="fileName">Name of the file</param>
        /// <returns>The complete file system path to the file</returns>
        public static string GetFilePath(string fileName)
        {
            return Path.Combine(Application.persistentDataPath, RECORDINGS_DIRECTORY, fileName);
        }


        /// <summary>
        /// Loads multiple point clouds from a *.ply file
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
    public static ArrayList LoadPointCloudsFromPLY(string filename, Matrix4x4 object_pose)
    {
    ArrayList clouds = new ArrayList();

    // Read pose of object while recording the sequence
    string teachingPosePath = filename.Substring(0, filename.LastIndexOf('\\'));
    Matrix4x4 teachingPoseInv = FileHandler.ReadMatrix(teachingPosePath + "\\object_pose.txt").inverse;

    // After this prefix, the number of vertices is specified
    const string length_prefix = "element vertex ";
    int idx = length_prefix.Length;

    // Initialize coordinates and colors to an empty list
    ArrayList points = new ArrayList();
    ArrayList colors = new ArrayList();

    // Points to the current position in the points and colors arrays that are being filled
    int curr = 0;
    try
    {
        // Is true iff the line containing 'end_header' has been consumed
        bool read_header = false;
        foreach(string line in System.IO.File.ReadLines(filename))
        {
            // Actually read points into the vector
            if (read_header)
            {
                // There are more vertices in the file than specified in the header
                if(curr >= points.Capacity)
                {
                    Debug.Log("There are more vertices than specified in the header");
                }
                try
                {
                    // Current cloud is read, load net one
                    if (line.TrimEnd().Equals("end_cloud"))
                    {
                        clouds.Add(new PointCloud(points, colors, object_pose, teachingPoseInv));
                        Debug.Log($"Read point clouds with {points.Count} points");
                        points = new ArrayList();
                        colors = new ArrayList();
                        curr = 0;
                        continue;
                    }
                    (Vector3? coordinates, Color color) = ParseLine(line.Trim());
                    if (coordinates == null)
                    {
                        Debug.Log("Skipped malformed line: " + line);
                        continue;
                    }
                    points.Add(coordinates);
                    colors.Add(color);
                    curr++;
                }
                catch (FormatException)
                {
                    Debug.Log("All point coordinates must be floats!");
                }
            }
            else
            {
                // Header ended
                if (line.Trim().Equals("end_header"))
                {
                    read_header = true;
                    continue;
                }
                // Line contains number of vertices
                if (line.Length > idx && line.Substring(0, idx).Equals(length_prefix))
                {
                    try
                    {
                        int n = Int32.Parse(line.Trim().Substring(idx));
                        points = new ArrayList(n);
                        colors = new ArrayList(n);
                        Debug.Log($"Reading a point cloud with {n} points");
                    }
                    catch (FormatException)
                    {
                        Debug.Log("Number of vertices must be an integer");
                    }
                }
            }
        }

        // Add the last point cloud (if any) after the loop ends
        //if (points != null && points.Count > 0)
        //{
        //    clouds.Add(new PointCloud(points, colors, object_pose, teachingPoseInv));
        //}
    }
    catch (IOException ex) // Could not open file, probably does not exist or don't have permission
    {
        Debug.LogError(ex.Message);
    }

    // Convert List<PointCloud> to ArrayList before returning
    //ArrayList result = new ArrayList(clouds);
    return clouds;
}

        /// <summary>
        /// Loads a single point cloud from a *.ply file
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public static void LoadPointsFromPLY(string path, out ArrayList points, out ArrayList colors)
        {
            points = null;
            colors = null;

            // After this prefix, the number of vertices is specified
            const string length_prefix = "element vertex ";
            int idx = length_prefix.Length;

            // Points to the current position in the points and colors arrays that are being filled
            int curr = 0;
            try
            {
                // Is true iff the line containing 'end_header' has been consumed
                bool read_header = false;
                foreach(string line in System.IO.File.ReadLines(path))
                {
                    // Actually read points into the vector
                    if (read_header)
                    {
                        // There are more vertices in the file than specified in the header
                        if(curr >= points.Capacity)
                        {
                            Debug.Log("There are more vertices than specified in the header");
                        }
                        try
                        {
                            (Vector3? coordinates, Color color) = ParseLine(line.Trim());
                            if (coordinates == null)
                            {
                                Debug.Log("Skipped malformed line: " + line);
                                continue;
                            }
                            points.Add(coordinates);
                            colors.Add(color);
                            curr++;
                        }
                        catch (FormatException)
                        {
                            Debug.Log("All point coordinates must be floats!");
                        }
                    }
                    else
                    {
                        // Header ended
                        if (line.Trim().Equals("end_header"))
                        {
                            read_header = true;
                            continue;
                        }
                        // Line contains number of vertices
                        if (line.Length > idx && line.Substring(0, idx).Equals(length_prefix))
                        {
                            try
                            {
                                int n = Int32.Parse(line.Trim().Substring(idx));
                                points = new ArrayList(n);
                                colors = new ArrayList(n);
                            }
                            catch (FormatException)
                            {
                                Debug.Log("Number of vertices must be an integer");
                            }
                        }
                    }
                }
            }
            catch (IOException ex) // Could not open file, probably does not exist or don't have permission
            {
                Debug.LogError(ex.Message);
            }
        }

        private static (Vector3?, Color) ParseLine(string line)
        {
            Color default_color = new Color(210f / 255f, 160f / 255f, 150f / 255f);
                        string[] elements = line.Split(' ');
            // Cannot parse malformed line
            if (elements.Length < 3) return (null, default_color);
            // x,y and z coordinates value of the point are the first three elements, seperated by space
            // the next three space-seperated elements are the r,g and b value
            float x = float.Parse(elements[0]);
            float y = float.Parse(elements[1]);
            float z = float.Parse(elements[2]);

            Color color = default_color;
            // use default color if RGB was not specified
            if (elements.Length >= 6)
            {
                float r = float.Parse(elements[3]) / 255f;
                float g = float.Parse(elements[4]) / 255f;
                float b = float.Parse(elements[5]) / 255f; 
                color = new Color(r, g, b);
            }



            return (new Vector3(x, y, z), color);
        }

        public static Matrix4x4 ReadMatrix(string path)
        {
            Matrix4x4 result = new Matrix4x4();
            int row = 0;
            foreach(string line in System.IO.File.ReadLines(path))
            {
                string[] elements = line.Split(' ');
                result[row, 0] = float.Parse(elements[0]);
                result[row, 1] = float.Parse(elements[1]);
                result[row, 2] = float.Parse(elements[2]);
                result[row, 3] = float.Parse(elements[3]);
                row++;
            }
            return result;
        }

    }

}
