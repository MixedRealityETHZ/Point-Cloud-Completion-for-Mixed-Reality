using System.Numerics;
using UnityEngine;
using System.Collections;

#if ENABLE_WINMD_SUPPORT
using Windows.Graphics.Imaging;
#endif

public class Frame
{
    public long timeStamp { get; set; }
    public System.Numerics.Matrix4x4 extrinsics { get; set; }
    public Frame(long timeStamp)
    {
        this.timeStamp = timeStamp;
    }
}

public class DepthFrame : Frame
{
    public ushort[] data { get; set; }

    public PointCloud pc;

    public DepthFrame(long ts, ushort[] data)
        : base(ts)
    {
        this.data = data;
    }
}

#if ENABLE_WINMD_SUPPORT
public class ColorFrame : Frame
{
    public ArrayList leftJoints { get; set;}
    public ArrayList rightJoints { get; set;}
    public SoftwareBitmap bitmap { get; set;}
    public System.Numerics.Matrix4x4 intrinsics { get; set; }

    public ColorFrame(long ts, SoftwareBitmap bmp)
        :base(ts) 
    {
        bitmap = bmp;
        leftJoints = new ArrayList();
        rightJoints = new ArrayList();
    }
}
#endif
