using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Windows.WebCam;
using System.Numerics;
using UnityEngine.Experimental.Rendering;
using Microsoft.MixedReality.Toolkit.Utilities;

#if ENABLE_WINMD_SUPPORT
using HL2UnityPlugin;
using Windows.Storage;
using Windows.Storage.Streams;
using StreamBuffer = Windows.Storage.Streams.Buffer;
using Windows.Media.Capture;
using Windows.Perception.Spatial;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;
using Windows.Graphics.Imaging;
using System.Runtime.InteropServices;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit.Input;
using System.Runtime.InteropServices.WindowsRuntime;
#endif

public class AnimationRecorder : MonoBehaviour
{   
    // follwing five classes are for saving the relevant values from the QR code.
    public GameObject pose_name_holder;

    private GameObject qrCodeObject;
    private GameObject qrCodeInfo;
    private GameObject qrcube;
    private GameObject IDnumber;
    public GameObject cube_pose;
    private string previous_ID;

    public GameObject logger;
    private DebugOutput dbg;
    private PointCloudCollection current_animation;
    public GameObject recordingButton;

    private PhotoCapture photoCaptureObject = null;
    private Texture2D targetTexture = null;
    private CameraParameters cameraParameters;
    private DateTime recordingStart;

    private bool recording = false;
    private string trackedObjectID;
    private static System.Random random = new System.Random();
    private UnityEngine.Matrix4x4 objectPose;
#if ENABLE_WINMD_SUPPORT
        private MediaCapture mediaCapture;
        private LowLagPhotoCapture photoCapture;
        private MediaFrameReader videoFrameReader;
        HL2ResearchMode researchMode;
        bool isCapturing;
        Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
        ArrayList colorFrames;
        float[] mappings;
#endif

    // Start is called before the first frame update
    void Start()
    {
        if (dbg == null)
        {
            dbg = logger.GetComponent<DebugOutput>();
        }
        dbg.Log("Starting AnimationRecorder");

        // We need this Gameobject to save the pose and id number of the corresponding QR code.
        // Essentially, these are the gameobjects that contain the relevant information that we need to 
        // save and pass on the recorder.

        try
        {   
            this.pose_name_holder = GameObject.Find("Pose_ID_Holder");
            //dbg.Log("qrCodeObject successfully defined");
        }
        catch
        {
            dbg.Log("game object not found !!!!!!");
        }
        try
        {
            //this.previous_ID = this.pose_name_holder.GetComponent<TextMesh>().text;
            this.previous_ID = "ID:";
            //dbg.Log("Id is also found successfully");
        }
        catch
        {
            dbg.Log("Text is not found!!!!!!");
        }

        #if ENABLE_WINMD_SUPPORT

            Windows.Storage.StorageFolder storageFolder = KnownFolders.Objects3D;
            string objectPath = $"{storageFolder.Path}";

            dbg.Log($"The images will be saved to: {objectPath}" );
        #endif

#if ENABLE_WINMD_SUPPORT
        IntPtr WorldOriginPtr = UnityEngine.XR.WindowsMR.WindowsMREnvironment.OriginSpatialCoordinateSystem;
        unityWorldOrigin = Marshal.GetObjectForIUnknown(WorldOriginPtr) as Windows.Perception.Spatial.SpatialCoordinateSystem;

        // Initialize camera
        // Define one array for color collection and an object of type PointCloudCollection for the P-clouds
        colorFrames = new ArrayList(); 
        InitCamera(); 
        isCapturing = false;
#endif
        current_animation = new PointCloudCollection();

        recording = false;

        try
        {
            //InitResearchMode();
        }
        catch (Exception e)
        {
            dbg.Log("Caught exception: " + e.ToString());
        }

    }

    void Update()
    {
        // We check if the QR ID has been changed, if changed, we call setObjectPose and 
        // SetObject.
        //dbg.Log($"Name is:  {this.pose_name_holder.GetComponent<TextMesh>().text}");

        this.pose_name_holder = GameObject.Find("Pose_ID_Holder");

        if(this.pose_name_holder.GetComponent<TextMesh>().text != this.previous_ID)
        {
            dbg.Log("New Qr detected in recorder!!");
            // Now we set the pose and the name of the recording to the detected values
            SetObject(this.pose_name_holder.GetComponent<TextMesh>().text);
            UnityEngine.Vector3 position = this.pose_name_holder.transform.position;
            //dbg.Log($"Position is: {position}");
            UnityEngine.Quaternion rotation = this.pose_name_holder.transform.rotation;
            //dbg.Log($"Rotation is: {rotation}");
            UnityEngine.Matrix4x4 Matrixholder = UnityEngine.Matrix4x4.TRS(position, rotation, UnityEngine.Vector3.one);
            SetObjectPose(Matrixholder); // Contains homog. Transformation of the qr code

            this.previous_ID = this.pose_name_holder.GetComponent<TextMesh>().text;
        }
    }

#if ENABLE_WINMD_SUPPORT
    // Initializes the rgb camera
    async Task InitCamera()
    {
        try
        {   
            // Access all available media frame sources and selects the first one (or default one)
            // that handles rgb images
            var frameSourceGroups = await MediaFrameSourceGroup.FindAllAsync();
            var selectedGroupObjects = frameSourceGroups.Select(group =>
               new
               {
                   sourceGroup = group,
                   colorSourceInfo = group.SourceInfos.FirstOrDefault((sourceInfo) =>
                   {
                       return sourceInfo.MediaStreamType == MediaStreamType.VideoPreview
                       && sourceInfo.SourceKind == MediaFrameSourceKind.Color;
                   })

               }).Where(t => t.colorSourceInfo != null)
               .FirstOrDefault();

            MediaFrameSourceGroup selectedGroup = selectedGroupObjects?.sourceGroup;
            MediaFrameSourceInfo colorSourceInfo = selectedGroupObjects?.colorSourceInfo;

            if (selectedGroup == null)
            {
                dbg.Log("Error: selectedGroup is null");
                return;
            }

            mediaCapture = new MediaCapture();
            var settings = new MediaCaptureInitializationSettings
            {
                SourceGroup = selectedGroup,
                SharingMode = MediaCaptureSharingMode.ExclusiveControl,
                MemoryPreference = MediaCaptureMemoryPreference.Cpu,
                StreamingCaptureMode = StreamingCaptureMode.Video
            };
            await mediaCapture.InitializeAsync(settings);

            var colorFrameSource = mediaCapture.FrameSources[colorSourceInfo.Id];
            var preferredFormat = colorFrameSource.SupportedFormats.Where(format =>
            {
                return format.VideoFormat.Width < 1080
                //&& format.Subtype == MediaEncodingSubtypes.Argb32
                && (int)Math.Round((double)format.FrameRate.Numerator / format.FrameRate.Denominator) == 30;

            }).FirstOrDefault();

            if (preferredFormat == null)
            {
                // Our desired format is not supported
                dbg.Log("Error: could not find format");
                return;
            }

            dbg.Log($"Using format with Subtype {preferredFormat.Subtype}");

            await colorFrameSource.SetFormatAsync(preferredFormat);

            mediaCapture.Failed += (MediaCapture sender, MediaCaptureFailedEventArgs errorEventArgs) =>
            {
                dbg.Log($"MediaCapture initialization failed: {errorEventArgs.Message}");
            };

            videoFrameReader = await mediaCapture.CreateFrameReaderAsync(colorFrameSource);
            videoFrameReader.FrameArrived += FrameArrived;
        }
        catch (Exception e)
        {
            dbg.Log(e.ToString());
        }
    }

    async void SaveSoftwareBitmapToFile(SoftwareBitmap softwareBitmap, StorageFile outputFile)
    {
        using (IRandomAccessStream stream = await outputFile.OpenAsync(FileAccessMode.ReadWrite))
        {
            // Create an encoder with the desired format
            BitmapEncoder encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, stream);

            // Set the software bitmap
            try 
            {
                softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                encoder.SetSoftwareBitmap(softwareBitmap);
            }
            catch (Exception e)
            {
                dbg.Log($"Caught exception trying to convert and set bitmap: '{e.ToString()}'");
            }

            encoder.IsThumbnailGenerated = true;

            try
            {
                await encoder.FlushAsync();
            }
            catch (Exception err)
            {
                const int WINCODEC_ERR_UNSUPPORTEDOPERATION = unchecked((int)0x88982F81);
                switch (err.HResult)
                {
                    case WINCODEC_ERR_UNSUPPORTEDOPERATION: 
                        // If the encoder does not support writing a thumbnail, then try again
                        // but disable thumbnail generation.
                        encoder.IsThumbnailGenerated = false;
                        break;
                    default:
                        throw;
                }
            }

            if (encoder.IsThumbnailGenerated == false)
            {
                try
                {
                    await encoder.FlushAsync();
                }
                catch(Exception e)
                {
                    dbg.Log($"Could not flush to image: '{e.ToString()}'");
                }

            }

        }
    }

    async void FrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
    {
        try 
        {
            using (var frame = sender.TryAcquireLatestFrame())
            {
                if (frame != null) 
                {
                    // see https://github.com/microsoft/psi/blob/cb2651f8e591c63d4a1fc8a16ad08ec7196338eb/Sources/MixedReality/Microsoft.Psi.MixedReality.UniversalWindows/MediaCapture/PhotoVideoCamera.cs#L529

                    // Compute pose
                    SpatialCoordinateSystem extrinsics = frame.CoordinateSystem;
                    System.Numerics.Matrix4x4 M = extrinsics.TryGetTransformTo(unityWorldOrigin) is System.Numerics.Matrix4x4 mat
                                        ? mat 
                                        : System.Numerics.Matrix4x4.Identity;

                    // Timestamp
                    long ticks = frame.SystemRelativeTime is TimeSpan s ? s.Ticks : -1;

                    // Get camera intrinsics
                    var intrinsics = frame.VideoMediaFrame.CameraIntrinsics;

                    // Constrict matrix from it
                    System.Numerics.Matrix4x4 projMat = System.Numerics.Matrix4x4.Identity;
                    // Focal lengths
                    projMat.M11 = intrinsics.FocalLength.X;
                    projMat.M22 = intrinsics.FocalLength.Y;
                    // Principal point
                    projMat.M13 = intrinsics.PrincipalPoint.X;
                    projMat.M23 = intrinsics.PrincipalPoint.Y;

                    using (var frameBitmap = frame.VideoMediaFrame.SoftwareBitmap)
                    {
                        if (frameBitmap == null)
                        {
                            dbg.Log("frameBitmap is null!");
                        }
                        // Copies bitmap to point cloud
                        ColorFrame colorFrame = new ColorFrame(ticks, SoftwareBitmap.Copy(frameBitmap));
                        colorFrame.extrinsics = M is System.Numerics.Matrix4x4 ext ? ext : System.Numerics.Matrix4x4.Identity;
                        colorFrame.intrinsics = projMat;
                        CaptureJointPositions(colorFrame);
                        colorFrames.Add(colorFrame);
                    }

                } else {
                    dbg.Log("frame is null");
                }
            }
        }
        catch(Exception e)
        {
            dbg.Log($"Caught exception in FrameArrived: '{e.ToString()}'");
        }
    }

    protected void CaptureJointPositions(ColorFrame cf)
    {
        // Get hand positions for segmentation
        MixedRealityPose pose;

        foreach (var jointName in Enum.GetNames(typeof(TrackedHandJoint)))
        {
            TrackedHandJoint joint = (TrackedHandJoint) Enum.Parse(typeof(TrackedHandJoint), jointName);
            if (HandJointUtils.TryGetJointPose(joint, Handedness.Right, out pose))
            {
                cf.rightJoints.Add(pose.Position);
            }
            else
            {
                cf.rightJoints.Add(UnityEngine.Vector3.zero);
            }

            if (HandJointUtils.TryGetJointPose(joint, Handedness.Left, out pose))
            {
                cf.leftJoints.Add(pose.Position);
            }
            else
            {
                cf.leftJoints.Add(UnityEngine.Vector3.zero);
            }
        }

    }

#endif

    // TODO: get 6DoF pose of tracked object
    // function gets called on button call
    public async void ToggleRecording()
    {
        // Write captured point cloud animation to disk
        if (recording)
        {
            try
            {
                string objectName;
                if (this.trackedObjectID != null)
                    objectName = this.trackedObjectID;
                else
                    objectName = "New-Object";

                recording = false;
#if ENABLE_WINMD_SUPPORT
                // Stop depth capture
                researchMode.StopDepthSensorLoop();

                // Stop video stream
                await videoFrameReader.StopAsync();
                var recordingEnd = DateTime.Now;
                SaveRecording(objectName);
                double seconds = (recordingEnd - recordingStart).TotalSeconds;
                var processingEnd = DateTime.Now;
                Windows.Storage.StorageFolder storageFolder = KnownFolders.Objects3D;
                dbg.Log($"Storage folder where data is saved: {storageFolder}");
                dbg.Log($"Done processing! Depth frame rate = {researchMode.GetPointCloudCount() / seconds}, RGB frame rate = {colorFrames.Count / seconds}");
                dbg.Log($"Processing time: {(processingEnd - recordingEnd).TotalSeconds}s = {(processingEnd - recordingEnd).TotalMinutes}min");

#endif
                current_animation = new PointCloudCollection();
            }
            catch (Exception e)
            {
                dbg.Log(e.ToString());
            }
        }
        else
        {
            recording = true;
            recordingStart = DateTime.Now;
#if ENABLE_WINMD_SUPPORT

            try 
            {
                // Start video
                var status = await videoFrameReader.StartAsync();
                if (status == MediaFrameReaderStartStatus.Success)
                {
                    dbg.Log("Successfully started mediaframereader!");
                }
                else
                {
                    dbg.Log($"Failed to start mediaframereader!, status = {status}");
                }
                InitResearchMode();
            }
            catch(Exception e)
            {
                dbg.Log(e.ToString());
            }
#endif
        }
    }


    public static string RandomString(int length)
    {
        const string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        return new string(Enumerable.Repeat(chars, length)
            .Select(s => s[random.Next(s.Length)]).ToArray());
    }

    private async void SaveRecording(string objectName)
    {
#if ENABLE_WINMD_SUPPORT
        try
        {
            Windows.Storage.StorageFolder storageFolder = KnownFolders.Objects3D;

            string objectPath = $"{storageFolder.Path}/{objectName}";
            string tutorialName = $"{RandomString(8)}-{RandomString(8)}-{RandomString(8)}-{RandomString(8)}";
            string tutorialPath = $"{objectPath}/{tutorialName}";
            string rgbPath = $"{tutorialPath}/rgb";
            string depthPath = $"{tutorialPath}/depth";

            // Make sure the directories for saving the data exists
            if (!Directory.Exists(objectPath))
            {
                Directory.CreateDirectory(objectPath);
                dbg.Log($"Directory created 01: objectPath {objectPath} \n");
            }
            else {
                dbg.Log($"DEBUGGED ERROR 01: directory does not exist: objectPath {objectPath} \n");
            }
            if (!Directory.Exists(tutorialPath))
            {
                Directory.CreateDirectory(tutorialPath);
                dbg.Log($"Directory created 02: tutorialPath {tutorialPath} \n");
            }
            else {
                dbg.Log($"DEBUGGED ERROR 02: directory does not exist: tutorialPath {tutorialPath} \n");
            }
            if (!Directory.Exists(rgbPath))
            {
                Directory.CreateDirectory(rgbPath);
                dbg.Log($"Directory created 03: rgbPath {rgbPath} \n");
            }
            else {
                dbg.Log($"DEBUGGED ERROR 03: directory does not exist: rgbPath {rgbPath} \n");
            }
            if (!Directory.Exists(depthPath))
            {
                Directory.CreateDirectory(depthPath);
                dbg.Log($"Directory created 04: depthPath {depthPath} \n");
            }
            else {
                dbg.Log($"DEBUGGED ERROR 04: directory does not exist: depthPath {depthPath} \n");
            }

            // Write object pose wrt unity coordinate system
            using (StreamWriter writer = new StreamWriter($"{depthPath}/object_pose.txt"))
            {   dbg.Log($"Write object pose wrt unity coordinate system: Using streamwriter");
                // Write pose
                writer.WriteLine($"{objectPose[0, 0]} {objectPose[0, 1]} {objectPose[0, 2]} {objectPose[0, 3]}");
                writer.WriteLine($"{objectPose[1, 0]} {objectPose[1, 1]} {objectPose[1, 2]} {objectPose[1, 3]}");
                writer.WriteLine($"{objectPose[2, 0]} {objectPose[2, 1]} {objectPose[2, 2]} {objectPose[2, 3]}");
                writer.WriteLine($"{objectPose[3, 0]} {objectPose[3, 1]} {objectPose[3, 2]} {objectPose[3, 3]}");
                dbg.Log($"Write object pose wrt unity coordinate system: Finished using streamwriter");
            }

            // Now we save the rgb images + hand_joint_pos_xyz + time stamp (to relate depth and color)
            for (int i = 0; i < colorFrames.Count; ++i)
            {   if (i == 0 || i == colorFrames.Count - 1) {
                dbg.Log($"colorFrames.Count = {colorFrames.Count}");
                }
                var frame = (ColorFrame)colorFrames[i];
                SoftwareBitmap bmp = SoftwareBitmap.Convert((SoftwareBitmap)(frame.bitmap),
                            BitmapPixelFormat.Rgba8, BitmapAlphaMode.Straight);

                Windows.Storage.StorageFolder obj = await storageFolder.GetFolderAsync(objectName);
                Windows.Storage.StorageFolder tut = await obj.GetFolderAsync(tutorialName);
                Windows.Storage.StorageFile file =
                    await(await tut.GetFolderAsync("rgb")).CreateFileAsync($"{i:D6}.png", Windows.Storage.CreationCollisionOption.ReplaceExisting);
                SaveSoftwareBitmapToFile(bmp, file);

                using (StreamWriter writer = new StreamWriter($"{rgbPath}/joints_{i:D6}.txt"))
                {   if (i == 0 || i == colorFrames.Count - 1) {
                        dbg.Log($"using streamWriter 2 writing pose");
                    }                
                    // Write pose
                    foreach (var joint in frame.leftJoints)
                    {
                        UnityEngine.Vector3 vecJoint = (UnityEngine.Vector3)joint;
                        writer.WriteLine($"{vecJoint.x} {vecJoint.y} {vecJoint.z}");
                    }

                    foreach (var joint in frame.rightJoints)
                    {
                        UnityEngine.Vector3 vecJoint = (UnityEngine.Vector3)joint;
                        writer.WriteLine($"{vecJoint.x} {vecJoint.y} {vecJoint.z}");
                    }
                    if (i == 0 || i == colorFrames.Count - 1) {
                        dbg.Log($"FINISHED using streamWriter 2 writing pose");
                    }

                }

                using (StreamWriter writer = new StreamWriter($"{rgbPath}/meta_{i:D6}.txt"))
                {
                    // Write time stamp to correlate rgb and depth images
                    writer.WriteLine($"{frame.timeStamp}");
                    if (i == 0 || i == colorFrames.Count - 1) {
                        dbg.Log($"using streamWriter 3 time stamp");
                    }

                    // Write pose
                    writer.WriteLine($"{frame.extrinsics.M11}, {frame.extrinsics.M12}, {frame.extrinsics.M13}, {frame.extrinsics.M14}");
                    writer.WriteLine($"{frame.extrinsics.M21}, {frame.extrinsics.M22}, {frame.extrinsics.M23}, {frame.extrinsics.M24}");
                    writer.WriteLine($"{frame.extrinsics.M31}, {frame.extrinsics.M32}, {frame.extrinsics.M33}, {frame.extrinsics.M34}");
                    writer.WriteLine($"{frame.extrinsics.M41}, {frame.extrinsics.M42}, {frame.extrinsics.M43}, {frame.extrinsics.M44}");

                    // Write intrinsics
                    writer.WriteLine($"{frame.intrinsics.M11}, {frame.intrinsics.M12}, {frame.intrinsics.M13}, {frame.intrinsics.M14}");
                    writer.WriteLine($"{frame.intrinsics.M21}, {frame.intrinsics.M22}, {frame.intrinsics.M23}, {frame.intrinsics.M24}");
                    writer.WriteLine($"{frame.intrinsics.M31}, {frame.intrinsics.M32}, {frame.intrinsics.M33}, {frame.intrinsics.M34}");
                    writer.WriteLine($"{frame.intrinsics.M41}, {frame.intrinsics.M42}, {frame.intrinsics.M43}, {frame.intrinsics.M44}");
                    if (i == 0 || i == colorFrames.Count - 1) {
                        dbg.Log($"FINISHED using streamWriter 3 time stamp");
                    }
                }
                dbg.Log($"Processed rgb frame {i + 1} of {colorFrames.Count}, number of depth images is {researchMode.GetPointCloudCount()}");
            }

            // Construct point clouds and write them to files
            for (uint i = 0; i < researchMode.GetPointCloudCount(); ++i)
            {
                long timeStamp = 0;
                float[] coordinates = researchMode.GetPointCloud(i, out timeStamp);
                PointCloud pc = new PointCloud(coordinates);
                float[] M = researchMode.GetDepthToWorld(i);
                pc.ExportToPLY($"{depthPath}/{i:D6}.ply");

                using (StreamWriter writer = new StreamWriter($"{depthPath}/meta_{i:D6}.txt"))
                {
                    // Write time stamp to correlate rgb and depth images
                    writer.WriteLine($"{timeStamp}");

                    // Write pose
                    writer.WriteLine($"{M[0]}, {M[1]}, {M[2]}, {M[3]}");
                    writer.WriteLine($"{M[4]}, {M[5]}, {M[6]}, {M[7]}");
                    writer.WriteLine($"{M[8]}, {M[9]}, {M[10]}, {M[11]}");
                    writer.WriteLine($"{M[12]}, {M[13]}, {M[14]}, {M[15]}");
                }
                dbg.Log($"Processed point cloud {i + 1}");

                //// TODO: remove again
                //Texture2D tex = new Texture2D(512, 512);
                //ushort[] depthImg = researchMode.GetDepthImage(i);
                //Debug.Log($"Length of array: {depthImg.Length}");
                //Debug.Log($"depthImg[200] = {depthImg[200]}");
                //Debug.Log($"depthImg[400] = {depthImg[400]}");
                //for(int j = 0; j < 512 * 512; ++j) 
                //{
                //    float val = depthImg[j] > 4090 ? 0 : ((float)depthImg[j]) / 4090f;
                //    tex.SetPixel(j / 512, j % 512, new Color(val, val, val));
                //}
                //tex.Apply();
                //byte[] bytes = ImageConversion.EncodeToPNG(tex);
                //File.WriteAllBytes($"{depthPath}/depth_image_{i}.png", bytes);
            }

            // Reset frames and mapping
            colorFrames = new ArrayList();
        }
        catch (Exception e)
        {   
            dbg.Log($"Error while saving images:\n '{e.ToString()}'");
            Debug.Log($"Error while saving images:\n '{e.ToString()}'");
        }
#endif
    }

    public void SetObject(string name)
    {
        // Don't set other object while recording
        try
        {
            dbg.Log($"Trying to set object");
            if (recording)
            {
                dbg.Log("Currently recording, setting object anyway");
            }
            dbg.Log($"Set object to {name}");
            this.trackedObjectID = name;
            recordingButton.SetActive(true);
        }
        catch (Exception e)
        {
            dbg.Log($"Caught exception in 'SetObject': {e.Message}");
        }
    }

    public void SetObjectPose(UnityEngine.Matrix4x4 pose)
    {
        try
        {
            this.objectPose = pose;
            //dbg.Log($"Pose of {this.trackedObjectID} is:\n{pose}");
        } catch (Exception e)
        {
            dbg.Log($"Caught exception in 'SetObjectPose': {e.Message}");
        }
    }

    public void UnsetObject()
    {
        // If no recording is running, don't allow user to start one
        // when there is no object for it to be associated with
        if (!recording)
        {
            this.trackedObjectID = null;
            recordingButton.SetActive(false);
        }
    }

    private void InitResearchMode()
    {
#if ENABLE_WINMD_SUPPORT
        try {
            researchMode = new HL2ResearchMode();

            researchMode.InitializeDepthSensor();
            researchMode.SetReferenceCoordinateSystem(unityWorldOrigin);
            researchMode.SetPointCloudDepthOffset(0);

            researchMode.StartDepthSensorLoop(true);
            //researchMode.StartSpatialCamerasFrontLoop();
            dbg.Log("Initialized research mode");
        } catch (Exception e) {
            dbg.Log(e.ToString());
        }
#endif
    }
}