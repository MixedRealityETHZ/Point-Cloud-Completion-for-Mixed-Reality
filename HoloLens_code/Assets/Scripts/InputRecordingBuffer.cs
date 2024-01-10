using System.Collections;
using System.Collections.Generic;
using Microsoft.MixedReality.Toolkit.Utilities;
using UnityEngine;

namespace Tutorials
{

    /// <summary>
    /// Simple class that is used to serialize and store the position and rotation of idividual joints or other objects.
    /// Note: This class takes the LOCAL position and rotation, as these are the relevant values for joints, since joints' positions and rotations always depend on their respective parent joints. 
    /// </summary>
    public class TransformData
    {
        public float posx;
        public float posy;
        public float posz;

        public float rotx;
        public float roty;
        public float rotz;
        public float rotw;

        public float scalex;
        public float scaley;
        public float scalez;

        public static TransformData ZeroIdentity()
        {
            return new TransformData(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        }

        public TransformData(Transform transform)
        {
            posx = transform.localPosition.x;
            posy = transform.localPosition.y;
            posz = transform.localPosition.z;

            rotx = transform.localRotation.x;
            roty = transform.localRotation.y;
            rotz = transform.localRotation.z;
            rotw = transform.localRotation.w;
            
            scalex = transform.localScale.x;
            scaley = transform.localScale.y;
            scalez = transform.localScale.z;
        }

        public TransformData(float posx,
                             float posy,
                             float posz,
                             float rotx,
                             float roty,
                             float rotz,
                             float rotw,
                             float scalex,
                             float scaley,
                             float scalez)
        {
            this.posx = posx;
            this.posy = posy;
            this.posz = posz;

            this.rotx = rotx;
            this.roty = roty;
            this.rotz = rotz;
            this.rotw = rotw;

            this.scalex = scalex;
            this.scaley = scaley;
            this.scalez = scalez;
        }

        /// <summary>
        /// Returns the position as a Vector3 object
        /// </summary>
        public Vector3 GetPosition()
        {
            return new Vector3(posx, posy, posz);
        }

        /// <summary>
        /// Gets the rotation as a Quaternion
        /// </summary>
        public Quaternion GetRotation()
        {
            return new Quaternion(rotx, roty, rotz, rotw);
        }

        /// <summary>
        /// Gets the scale as a Vector3 object
        /// </summary>
        public Vector3 GetScale()
        {
            return new Vector3(scalex, scaley, scalez);
        }

        /// <summary>
        /// Converts the TransForm to a human readable string.
        /// </summary>
        /// <returns>
        /// A <see cref="System.String" /> that represents this instance.
        /// </returns>
        public override string ToString()
        {
            return string.Format("[{0},{1},{2}][{3},{4},{5},{6}]", posx, posy, posz, rotx, roty, rotz, rotw);
        }
    }


    /// <summary>
    /// Container used to efficiently store a sequence of input animation keyframes while recording
    /// </summary>
    public class InputRecordingBuffer : IEnumerable<InputRecordingBuffer.Keyframe>
    {
        /// <summary>
        /// The input state for a single frame
        /// </summary>
        public class Keyframe
        {
            public float Time { get; set; }
            public bool LeftTracked { get; set; }
            public bool RightTracked { get; set; }
            public bool LeftPinch { get; set; }
            public bool RightPinch { get; set; }
            public MixedRealityPose CameraPose { get; set; }
            public MixedRealityPose GazePose { get; set; }
            public Dictionary<TrackedHandJoint, MixedRealityPose> LeftJoints { get; set; }
            public Dictionary<TrackedHandJoint, MixedRealityPose> RightJoints { get; set; }

            public Dictionary<TrackedHandJoint, TransformData> LeftJointsTransformData { get; set; }
            public Dictionary<TrackedHandJoint, TransformData> RightJointsTransformData { get; set; }
            /// <summary>
            /// Holds transformation data for each object recorded
            /// </summary>
            public Dictionary<string, TransformData> ObjectsTransformData { get; set; }

            /// <summary>
            /// Initialize the data structures to hold the joint/object transformations for a time-specific keyframe
            /// </summary>
            /// <param name="time"></param>
            public Keyframe(float time)
            {
                Time = time;
                LeftJoints = new Dictionary<TrackedHandJoint, MixedRealityPose>();
                RightJoints = new Dictionary<TrackedHandJoint, MixedRealityPose>();
                LeftJointsTransformData = new Dictionary<TrackedHandJoint, TransformData>();
                RightJointsTransformData = new Dictionary<TrackedHandJoint, TransformData>();
                ObjectsTransformData = new Dictionary<string, TransformData>();
            }
        }

        /// <summary>
        /// The time of the first keyframe in the buffer
        /// </summary>
        public float StartTime
        {
            get
            {
                return keyframes.Peek().Time;
            }
        }

        private Keyframe currentKeyframe;
        private Queue<Keyframe> keyframes;

        /// <summary>
        /// Default constructor
        /// </summary>
        public InputRecordingBuffer() => keyframes = new Queue<Keyframe>();

        /// <summary>
        /// Removes all keyframes from the buffer
        /// </summary>
        public void Clear()
        {
            keyframes.Clear();
            currentKeyframe = null;
        }

        /// <summary>
        /// Removes all keyframes before a given time
        /// </summary>
        public void RemoveBeforeTime(float time)
        {
            while (keyframes.Count > 0 && keyframes.Peek().Time < time)
            {
                keyframes.Dequeue();
            }
        }

        /// <summary>
        /// Sets the camera pose to be stored in the newest keyframe
        /// </summary>
        public void SetCameraPose(MixedRealityPose pose) => currentKeyframe.CameraPose = pose;

        /// <summary>
        /// Sets the eye gaze pose to be stored in the newest keyframe
        /// </summary>
        public void SetGazePose(MixedRealityPose pose) => currentKeyframe.GazePose = pose;

        /// <summary>
        /// Sets the state of a given hand to be stored in the newest keyframe
        /// </summary>
        public void SetHandState(Handedness handedness, bool tracked, bool pinching)
        {
            if (handedness == Handedness.Left)
            {
                currentKeyframe.LeftTracked = tracked;
                currentKeyframe.LeftPinch = pinching;
            }
            else
            {
                currentKeyframe.RightTracked = tracked;
                currentKeyframe.RightPinch = pinching;
            }
        }

        /// <summary>
        /// Sets the pose of a given joint to be stored in the newest keyframe
        /// </summary>
        public void SetJointPose(Handedness handedness, TrackedHandJoint joint, MixedRealityPose pose)
        {
            if (handedness == Handedness.Left)
            {
                currentKeyframe.LeftJoints.Add(joint, pose);
            }
            else
            {
                currentKeyframe.RightJoints.Add(joint, pose);
            }
        }

        /// <summary>
        /// Sets the transform data of a given joint to be stored in the newest keyframe
        /// </summary>
        public void SetJointTransform(Handedness handedness, TrackedHandJoint joint, Transform transform)
        {
            if (handedness == Handedness.Left)
            {
                currentKeyframe.LeftJointsTransformData.Add(joint, new TransformData(transform));
            }
            else
            {
                currentKeyframe.RightJointsTransformData.Add(joint, new TransformData(transform));
            }
        }

        public void SetObjectState(Transform objectTransform)
        {
            currentKeyframe.ObjectsTransformData.Add(objectTransform.name, new TransformData(objectTransform));
        }

        /// <summary>
        /// Creates a new, empty keyframe with a given time at the end of the buffer
        /// </summary>
        /// <returns>The index of the new keyframe</returns>
        public int NewKeyframe(float time)
        {
            currentKeyframe = new Keyframe(time);
            keyframes.Enqueue(currentKeyframe);

            return keyframes.Count - 1;
        }

        /// <summary>
        /// Returns a sequence of all keyframes in the buffer
        /// </summary>
        public IEnumerator<Keyframe> GetEnumerator() => keyframes.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}