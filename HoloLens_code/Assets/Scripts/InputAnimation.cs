using Microsoft.MixedReality.Toolkit.Utilities;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace Tutorials
{

    /// <summary>
    /// A used-defined marker on the input animation timeline.
    /// </summary>
    [Serializable]
    public class InputAnimationMarker
    {
        /// <summary>
        /// Placement of the marker relative to the input animation start time.
        /// </summary>
        public float time = 0.0f;

        /// <summary>
        /// Custom name of the marker.
        /// </summary>
        public string name = "";
    }

    /// <summary>
    /// Contains a set of animation curves that describe motion of camera and hands.
    /// </summary>
    [System.Serializable]
    public class InputAnimation
    {
        protected static readonly int jointCount = Enum.GetNames(typeof(TrackedHandJoint)).Length;

        /// <summary>
        /// Arbitrarily large weight for representing a boolean value in float curves.
        /// </summary>
        private const float BoolOutWeight = 1.0e6f;

        /// <summary>
        /// Maximum duration of all animations curves.
        /// </summary>
        [SerializeField]
        private float duration = 0.0f;

        /// <summary>
        /// Name of the animation visible to the user
        /// </summary>
        public string description = "Unnamed scene";

        /// <summary>
        /// Maximum duration of all animations curves.
        /// </summary>
        public float Duration
        {
            get
            {
                if (duration > 0.0f) return duration;

                ComputeDuration();

                return duration;
            }
        }

        /// <summary>
        /// Name of the animation visible to the user
        /// </summary>
        /*public string Description { get
            {
                return description;
            }
            protected internal set
            {
                this.description = value;
            }
        }*/

        /// <summary>
        /// Class that contains all animation curves for one joint / object (position and rotation)
        /// </summary>
        public class PoseCurves
        {
            public AnimationCurve PositionX = new AnimationCurve();
            public AnimationCurve PositionY = new AnimationCurve();
            public AnimationCurve PositionZ = new AnimationCurve();
            public AnimationCurve RotationX = new AnimationCurve();
            public AnimationCurve RotationY = new AnimationCurve();
            public AnimationCurve RotationZ = new AnimationCurve();
            public AnimationCurve RotationW = new AnimationCurve();
            public AnimationCurve ScaleX = new AnimationCurve();
            public AnimationCurve ScaleY = new AnimationCurve();
            public AnimationCurve ScaleZ = new AnimationCurve();

            public void AddKey(float time, MixedRealityPose pose)
            {
                AddFloatKey(PositionX, time, pose.Position.x);
                AddFloatKey(PositionY, time, pose.Position.y);
                AddFloatKey(PositionZ, time, pose.Position.z);

                AddFloatKey(RotationX, time, pose.Rotation.x);
                AddFloatKey(RotationY, time, pose.Rotation.y);
                AddFloatKey(RotationZ, time, pose.Rotation.z);
                AddFloatKey(RotationW, time, pose.Rotation.w);
            }

            public void AddKey(float time, TransformData transformData)
            {
                AddFloatKey(PositionX, time, transformData.posx);
                AddFloatKey(PositionY, time, transformData.posy);
                AddFloatKey(PositionZ, time, transformData.posz);

                AddFloatKey(RotationX, time, transformData.rotx);
                AddFloatKey(RotationY, time, transformData.roty);
                AddFloatKey(RotationZ, time, transformData.rotz);
                AddFloatKey(RotationW, time, transformData.rotw);

                AddFloatKey(ScaleX, time, transformData.scalex);
                AddFloatKey(ScaleY, time, transformData.scaley);
                AddFloatKey(ScaleZ, time, transformData.scalez);
            }

            /// <summary>
            /// Optimizes the set of curves.
            /// </summary>
            /// <param name="positionThreshold">The maximum permitted error between the positions of the old and new curves, in units.</param>
            /// <param name="rotationThreshold">The maximum permitted error between the rotations of the old and new curves, in degrees.</param>
            /// <param name="partitionSize">The size of the partitions of the curves that will be optimized independently. Larger values will optimize the curves better, but may take longer.</param>
            public void Optimize(float positionThreshold, float rotationThreshold, int partitionSize)
            {
                OptimizePositionCurve(ref PositionX, ref PositionY, ref PositionZ, positionThreshold, partitionSize);
                OptimizeRotationCurve(ref RotationX, ref RotationY, ref RotationZ, ref RotationW, rotationThreshold, partitionSize);
            }

            public TransformData Evaluate(float time)
            {
                var transformData = new TransformData(PositionX.Evaluate(time),
                                                PositionY.Evaluate(time),
                                                PositionZ.Evaluate(time),
                                                RotationX.Evaluate(time),
                                                RotationY.Evaluate(time),
                                                RotationZ.Evaluate(time),
                                                RotationW.Evaluate(time),
                                                ScaleX.Evaluate(time),
                                                ScaleY.Evaluate(time),
                                                ScaleZ.Evaluate(time));


                return transformData;
            }
        }

        private class RayCurves
        {
            public AnimationCurve OriginX = new AnimationCurve();
            public AnimationCurve OriginY = new AnimationCurve();
            public AnimationCurve OriginZ = new AnimationCurve();
            public AnimationCurve DirectionX = new AnimationCurve();
            public AnimationCurve DirectionY = new AnimationCurve();
            public AnimationCurve DirectionZ = new AnimationCurve();

            public void AddKey(float time, Ray ray)
            {
                AddVectorKey(OriginX, OriginY, OriginZ, time, ray.origin);
                AddVectorKey(DirectionX, DirectionY, DirectionZ, time, ray.direction);
            }

            /// <summary>
            /// Optimizes the set of curves.
            /// </summary>
            /// <param name="originThreshold">The maximum permitted error between the origins of the old and new curves, in units.</param>
            /// <param name="directionThreshold">The maximum permitted error between the directions of the old and new curves, in degrees.</param>
            /// <param name="partitionSize">The size of the partitions of the curves that will be optimized independently. Larger values will optimize the curves better, but may take longer.</param>
            public void Optimize(float originThreshold, float directionThreshold, int partitionSize)
            {
                OptimizePositionCurve(ref OriginX, ref OriginY, ref OriginZ, originThreshold, partitionSize);
                OptimizeDirectionCurve(ref DirectionX, ref DirectionY, ref DirectionZ, directionThreshold, partitionSize);
            }

            public Ray Evaluate(float time)
            {
                float ox = OriginX.Evaluate(time);
                float oy = OriginY.Evaluate(time);
                float oz = OriginZ.Evaluate(time);
                float dx = DirectionX.Evaluate(time);
                float dy = DirectionY.Evaluate(time);
                float dz = DirectionZ.Evaluate(time);

                var ray = new Ray();

                ray.origin = new Vector3(ox, oy, oz);
                ray.direction = new Vector3(dx, dy, dz);
                ray.direction.Normalize();

                return ray;
            }
        }

        internal class CompareMarkers : IComparer<InputAnimationMarker>
        {
            public int Compare(InputAnimationMarker a, InputAnimationMarker b)
            {
                return a.time.CompareTo(b.time);
            }
        }

        [SerializeField]
        public AnimationCurve handTrackedCurveLeft;
        [SerializeField]
        public AnimationCurve handTrackedCurveRight;
        [SerializeField]
        private AnimationCurve handPinchCurveLeft;
        [SerializeField]
        private AnimationCurve handPinchCurveRight;
        [SerializeField]
        private Dictionary<TrackedHandJoint, PoseCurves> handJointCurvesLeft;
        [SerializeField]
        private Dictionary<TrackedHandJoint, PoseCurves> handJointCurvesRight;
        [SerializeField]
        private PoseCurves cameraCurves;
        [SerializeField]
        private PoseCurves gazeCurves;
        [SerializeField]
        public Dictionary<string, PoseCurves> objectCurves;

        /// <summary>
        /// Whether the animation has hand state and joint curves
        /// </summary>
        public bool HasHandData { get; private set; } = false;
        /// <summary>
        /// Whether the animation has camera pose curves
        /// </summary>
        public bool HasCameraPose { get; private set; } = false;
        /// <summary>
        /// Whether the animation has eye gaze curves
        /// </summary>
        public bool HasEyeGaze { get; private set; } = false;

        /// <summary>
        /// Number of markers in the animation.
        /// </summary>
        [SerializeField]
        private List<InputAnimationMarker> markers;
        /// <summary>
        /// Number of markers in the animation.
        /// </summary>
        public int markerCount => markers.Count;

        /// <summary>
        /// Default constructor
        /// </summary>
        public InputAnimation()
        {
            handTrackedCurveLeft = new AnimationCurve();
            handTrackedCurveRight = new AnimationCurve();
            handPinchCurveLeft = new AnimationCurve();
            handPinchCurveRight = new AnimationCurve();
            handJointCurvesLeft = new Dictionary<TrackedHandJoint, PoseCurves>();
            handJointCurvesRight = new Dictionary<TrackedHandJoint, PoseCurves>();
            cameraCurves = new PoseCurves();
            gazeCurves = new PoseCurves();
            objectCurves = new Dictionary<string, PoseCurves>();
            markers = new List<InputAnimationMarker>();
        }

        /// <summary>
        /// Add a keyframe for the tracking state of a hand.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        public void AddHandStateKey(float time, Handedness handedness, bool isTracked, bool isPinching)
        {
            if (handedness == Handedness.Left)
            {
                AddHandStateKey(time, isTracked, isPinching, handTrackedCurveLeft, handPinchCurveLeft);
            }
            else if (handedness == Handedness.Right)
            {
                AddHandStateKey(time, isTracked, isPinching, handTrackedCurveRight, handPinchCurveRight);
            }
        }

        /// <summary>
        /// Add a keyframe for one hand joint.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        public void AddHandJointKey(float time, Handedness handedness, TrackedHandJoint joint, MixedRealityPose jointPose, float positionThreshold, float rotationThreshold)
        {
            if (handedness == Handedness.Left)
            {
                AddHandJointKey(time, joint, jointPose, handJointCurvesLeft, positionThreshold, rotationThreshold);
            }
            else if (handedness == Handedness.Right)
            {
                AddHandJointKey(time, joint, jointPose, handJointCurvesRight, positionThreshold, rotationThreshold);
            }
        }

        /// <summary>
        /// Add a keyframe for the camera transform.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        public void AddCameraPoseKey(float time, MixedRealityPose cameraPose, float positionThreshold, float rotationThreshold)
        {
            AddPoseKeyFiltered(cameraCurves, time, cameraPose, positionThreshold, rotationThreshold);

            duration = Mathf.Max(duration, time);
        }

        /// <summary>
        /// Add a user-defined marker.
        /// </summary>
        public void AddMarker(InputAnimationMarker marker)
        {
            int index = FindMarkerInterval(marker.time) + 1;
            markers.Insert(index, marker);
        }

        /// <summary>
        /// Remove the user-defined marker at the given index.
        /// </summary>
        public void RemoveMarker(int index)
        {
            markers.RemoveAt(index);
        }

        /// <summary>
        /// Change the time of the marker at the given index.
        /// </summary>
        public void SetMarkerTime(int index, float time)
        {
            var marker = markers[index];
            markers.RemoveAt(index);

            int newIndex = FindMarkerInterval(time) + 1;
            marker.time = time;
            markers.Insert(newIndex, marker);
        }

        /// <summary>
        /// Remove all keyframes from all animation curves.
        /// </summary>
        public void Clear()
        {
            foreach (var curve in GetAllAnimationCurves())
            {
                curve.keys = new Keyframe[0];
            }
        }

        /// <summary>
        /// Remove all keyframes from all animation curves with time values before the given cutoff time.
        /// </summary>
        /// <remarks>
        /// <para>If keyframes exists before the cutoff time then one preceding keyframe will be retained,
        /// so that interpolation at the cutoff time yields the same result.</para>
        /// </remarks>
        [Obsolete("Cutoff is achieved in InputRecordingBuffer")]
        public void CutoffBeforeTime(float time)
        {
            foreach (var curve in GetAllAnimationCurves())
            {
                CutoffBeforeTime(curve, time);
            }
        }

        public async Task<byte[]> ToBinary()
        {
            MemoryStream stream = new MemoryStream();
            await this.ToStreamAsync(stream);
            return stream.ToArray();
        }

        /// <summary>
        /// Serialize animation data into a stream.
        /// </summary>
        public void ToStream(Stream stream)
        {
            var writer = new BinaryWriter(stream);

            InputAnimationSerializationUtils.WriteHeader(writer);
            writer.Write(HasCameraPose);
            writer.Write(HasHandData);
            writer.Write(HasEyeGaze);

            var defaultCurves = new PoseCurves();

            if (HasCameraPose)
            {
                PoseCurvesToStream(writer, cameraCurves);
            }

            if (HasHandData)
            {
                InputAnimationSerializationUtils.WriteBoolCurve(writer, handTrackedCurveLeft);
                InputAnimationSerializationUtils.WriteBoolCurve(writer, handTrackedCurveRight);
                InputAnimationSerializationUtils.WriteBoolCurve(writer, handPinchCurveLeft);
                InputAnimationSerializationUtils.WriteBoolCurve(writer, handPinchCurveRight);

                for (int i = 0; i < jointCount; ++i)
                {
                    if (!handJointCurvesLeft.TryGetValue((TrackedHandJoint)i, out var curves))
                    {
                        curves = defaultCurves;
                    }
                    PoseCurvesToStream(writer, curves);
                }


                for (int i = 0; i < jointCount; ++i)
                {
                    if (!handJointCurvesRight.TryGetValue((TrackedHandJoint)i, out var curves))
                    {
                        curves = defaultCurves;
                    }
                    PoseCurvesToStream(writer, curves);
                }

            }

            if (HasEyeGaze)
            {
                PoseCurvesToStream(writer, gazeCurves);
            }

            ObjectCurvesToStream(writer, objectCurves);
            
            InputAnimationSerializationUtils.WriteMarkerList(writer, markers);
        }

        /// <summary>
        /// Serialize animation data into a stream asynchronously.
        /// </summary>
        public async Task ToStreamAsync(Stream stream, Action callback = null)
        {
            await Task.Run(() => ToStream(stream));

            callback?.Invoke();
        }

        /// <summary>
        /// Evaluate hand tracking state at the given time.
        /// </summary>
        public void EvaluateHandState(float time, Handedness handedness, out bool isTracked, out bool isPinching)
        {
            if (!HasHandData)
            {
                isTracked = false;
                isPinching = false;
            }

            if (handedness == Handedness.Left)
            {
                EvaluateHandState(time, handTrackedCurveLeft, handPinchCurveLeft, out isTracked, out isPinching);
            }
            else if (handedness == Handedness.Right)
            {
                EvaluateHandState(time, handTrackedCurveRight, handPinchCurveRight, out isTracked, out isPinching);
            }
            else
            {
                isTracked = false;
                isPinching = false;
            }
        }

        /// <summary>
        /// Find an index i in the sorted events list, such that events[i].time &lt;= time &lt; events[i+1].time.
        /// </summary>
        /// <returns>
        /// 0 &lt;= i &lt; eventCount if a full interval could be found.
        /// -1 if time is less than the first event time.
        /// eventCount-1 if time is greater than the last event time.
        /// </returns>
        /// <remarks>
        /// Uses binary search.
        /// </remarks>
        public int FindMarkerInterval(float time)
        {
            int lowIdx = -1;
            int highIdx = markers.Count;
            while (lowIdx < highIdx - 1)
            {
                int midIdx = (lowIdx + highIdx) >> 1;
                if (time >= markers[midIdx].time)
                {
                    lowIdx = midIdx;
                }
                else
                {
                    highIdx = midIdx;
                }
            }
            return lowIdx;
        }

        /// <summary>
        /// Evaluate joint pose at the given time.
        /// </summary>
        public TransformData EvaluateHandJoint(float time, Handedness handedness, TrackedHandJoint joint)
        {
            if (!HasHandData)
            {
                return TransformData.ZeroIdentity();
            }

            if (handedness == Handedness.Left)
            {
                return EvaluateHandJoint(time, joint, handJointCurvesLeft);
            }
            else if (handedness == Handedness.Right)
            {
                return EvaluateHandJoint(time, joint, handJointCurvesRight);
            }
            else
            {
                return TransformData.ZeroIdentity();
            }
        }

        /// <summary>
        /// Evaluate the eye gaze pose at the given time.
        /// </summary>
        public TransformData EvaluateEyeGaze(float time)
        {
            if (!HasEyeGaze)
            {
                return TransformData.ZeroIdentity();
            }

            return gazeCurves.Evaluate(time);
        }

        public TransformData EvaluateCameraPose(float time)
        {
            if (!HasCameraPose)
            {
                return TransformData.ZeroIdentity();
            }

            return cameraCurves.Evaluate(time);
        }


        /// <summary>
        /// Generates an input animation from the contents of a recording buffer.
        /// </summary>
        /// <param name="recordingBuffer">The buffer to convert to an animation</param>
        public static InputAnimation FromRecordingBuffer(InputRecordingBuffer recordingBuffer)
        {
            var animation = new InputAnimation();
            float startTime = recordingBuffer.StartTime;

            animation.HasHandData = true;

            foreach (var keyframe in recordingBuffer)
            {
                float localTime = keyframe.Time - startTime;


                AddBoolKeyIfChanged(animation.handTrackedCurveLeft, localTime, keyframe.LeftTracked);
                AddBoolKeyIfChanged(animation.handTrackedCurveRight, localTime, keyframe.RightTracked);
                AddBoolKeyIfChanged(animation.handPinchCurveLeft, localTime, keyframe.LeftPinch);
                AddBoolKeyIfChanged(animation.handPinchCurveRight, localTime, keyframe.RightPinch);

                foreach (var joint in (TrackedHandJoint[])Enum.GetValues(typeof(TrackedHandJoint)))
                {
                    AddJointPoseKeys(animation.handJointCurvesLeft, keyframe.LeftJointsTransformData, joint, localTime);
                    AddJointPoseKeys(animation.handJointCurvesRight, keyframe.RightJointsTransformData, joint, localTime);
                }
                foreach(var objectData in keyframe.ObjectsTransformData)
                {
                    AddObjectPoseKeys(animation.objectCurves, objectData.Value, objectData.Key, localTime);
                }
            }

            animation.ComputeDuration();

            return animation;

            void AddBoolKeyIfChanged(AnimationCurve curve, float time, bool value)
            {
                if (curve.length > 0 && (curve[curve.length - 1].value > 0.5f) == value)
                {
                    return;
                }

                AddBoolKey(curve, time, value);
            }

            void AddJointPoseKeys(Dictionary<TrackedHandJoint, PoseCurves> jointCurves, Dictionary<TrackedHandJoint, TransformData> jointsTransformData, TrackedHandJoint joint, float time)
            {
                if (!jointsTransformData.TryGetValue(joint, out var transformData))
                {
                    return;
                }

                if (!jointCurves.TryGetValue(joint, out var curves))
                {
                    curves = new PoseCurves();
                    jointCurves.Add(joint, curves);
                }

                curves.AddKey(time, transformData);
            }

            void AddObjectPoseKeys(Dictionary<string, PoseCurves> objectCurves, TransformData objectTransform, string objectName, float time)
            {
                if (!objectCurves.TryGetValue(objectName, out var curves))
                {
                    curves = new PoseCurves();
                    objectCurves.Add(objectName, curves);
                }
                curves.AddKey(time, objectTransform);
            }
        }

        /// <summary>
        /// Deserializes animation data from a stream.
        /// </summary>
        public static InputAnimation FromStream(Stream stream)
        {
            var animation = new InputAnimation();
            var reader = new BinaryReader(stream);

            InputAnimationSerializationUtils.ReadHeader(reader);

            animation.HasCameraPose = reader.ReadBoolean();
            animation.HasHandData = reader.ReadBoolean();
            animation.HasEyeGaze = reader.ReadBoolean();

            if (animation.HasCameraPose)
            {
                PoseCurvesFromStream(reader, animation.cameraCurves);
            }

            if (animation.HasHandData)
            {
                InputAnimationSerializationUtils.ReadBoolCurve(reader, animation.handTrackedCurveLeft);
                InputAnimationSerializationUtils.ReadBoolCurve(reader, animation.handTrackedCurveRight);
                InputAnimationSerializationUtils.ReadBoolCurve(reader, animation.handPinchCurveLeft);
                InputAnimationSerializationUtils.ReadBoolCurve(reader, animation.handPinchCurveRight);

                for (int i = 0; i < jointCount; ++i)
                {
                    if (!animation.handJointCurvesLeft.TryGetValue((TrackedHandJoint)i, out var curves))
                    {
                        curves = new PoseCurves();
                        animation.handJointCurvesLeft.Add((TrackedHandJoint)i, curves);
                    }

                    PoseCurvesFromStream(reader, curves);
                }


                for (int i = 0; i < jointCount; ++i)
                {
                    if (!animation.handJointCurvesRight.TryGetValue(key: (TrackedHandJoint)i, out var curves))
                    {
                        curves = new PoseCurves();
                        animation.handJointCurvesRight.Add((TrackedHandJoint)i, curves);
                    }

                    PoseCurvesFromStream(reader, curves);
                }

            }

            if (animation.HasEyeGaze)
            {
                PoseCurvesFromStream(reader, animation.gazeCurves);
            }

            ObjectCurvesFromStream(reader, animation.objectCurves);


            InputAnimationSerializationUtils.ReadMarkerList(reader, animation.markers);
            animation.ComputeDuration();

            return animation;
        }

        /// <summary>
        /// Deserialize animation data from a stream asynchronously.
        /// </summary>
        public static async Task<InputAnimation> FromStreamAsync(Stream stream, Action callback = null)
        {
            var result = await Task.Run(() => FromStream(stream));

            callback?.Invoke();

            return result;
        }

        /// <summary>
        /// Add a keyframe for the tracking state of a hand.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimation")]
        private void AddHandStateKey(float time, bool isTracked, bool isPinching, AnimationCurve trackedCurve, AnimationCurve pinchCurve)
        {
            AddBoolKeyFiltered(trackedCurve, time, isTracked);
            AddBoolKeyFiltered(pinchCurve, time, isPinching);

            duration = Mathf.Max(duration, time);
        }


        /// <summary>
        /// Add a keyframe for one hand joint.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        private void AddHandJointKey(float time, TrackedHandJoint joint, MixedRealityPose jointPose, Dictionary<TrackedHandJoint, PoseCurves> jointCurves, float positionThreshold, float rotationThreshold)
        {
            if (!jointCurves.TryGetValue(joint, out var curves))
            {
                curves = new PoseCurves();
                jointCurves.Add(joint, curves);
            }

            AddPoseKeyFiltered(curves, time, jointPose, positionThreshold, rotationThreshold);

            duration = Mathf.Max(duration, time);
        }

        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        private void CutoffBeforeTime(AnimationCurve curve, float time)
        {
            // Keep the keyframe before the cutoff time to ensure correct value at the beginning
            int idx0 = FindKeyframeInterval(curve, time);
            if (idx0 > 0)
            {
                var newKeys = new Keyframe[curve.keys.Length - idx0];
                for (int i = 0; i < newKeys.Length; ++i)
                {
                    newKeys[i] = curve.keys[idx0 + i];
                }
                curve.keys = newKeys;
            }
        }

        /// <summary>
        /// Make sure the pose animation curves for the given hand joint exist.
        /// </summary>
        [Obsolete("Unused")]
        private PoseCurves CreateHandJointCurves(Handedness handedness, TrackedHandJoint joint)
        {
            if (handedness == Handedness.Left)
            {
                if (!handJointCurvesLeft.TryGetValue(joint, out var curves))
                {
                    curves = new PoseCurves();
                    handJointCurvesLeft.Add(joint, curves);
                }
                return curves;
            }
            else if (handedness == Handedness.Right)
            {
                if (!handJointCurvesRight.TryGetValue(joint, out var curves))
                {
                    curves = new PoseCurves();
                    handJointCurvesRight.Add(joint, curves);
                }
                return curves;
            }
            return null;
        }

        /// <summary>
        /// Get animation curves for the pose of the given hand joint, if they exist.
        /// </summary>
        //[Obsolete("Use EvaluateHandJoint to get joint pose data")]
        public bool TryGetHandJointCurves(Handedness handedness, TrackedHandJoint joint, out PoseCurves curves)
        {
            if (handedness == Handedness.Left)
            {
                return handJointCurvesLeft.TryGetValue(joint, out curves);
            }
            else if (handedness == Handedness.Right)
            {
                return handJointCurvesRight.TryGetValue(joint, out curves);
            }
            curves = null;
            return false;
        }

        public void ComputeDuration()
        {
            duration = 0.0f;
            foreach (var curve in GetAllAnimationCurves())
            {
                float curveDuration = (curve.length > 0 ? curve.keys[curve.length - 1].time : 0.0f);
                duration = Mathf.Max(duration, curveDuration);
            }
        }

        /// <summary>
        /// Evaluate hand tracking state at the given time.
        /// </summary>
        private void EvaluateHandState(float time, AnimationCurve trackedCurve, AnimationCurve pinchCurve, out bool isTracked, out bool isPinching)
        {
            isTracked = (trackedCurve.Evaluate(time) > 0.5f);
            isPinching = (pinchCurve.Evaluate(time) > 0.5f);
        }

        /// <summary>
        /// Evaluate joint pose at the given time.
        /// </summary>
        private TransformData EvaluateHandJoint(float time, TrackedHandJoint joint, Dictionary<TrackedHandJoint, PoseCurves> jointCurves)
        {
            if (jointCurves.TryGetValue(joint, out var curves))
            {
                return curves.Evaluate(time);
            }
            else
            {
                // Zero Identity Transform (pos 0, rot 0)
                return TransformData.ZeroIdentity();
            }
        }

        public TransformData EvaluateObject(float time, string name)
        {
            if (objectCurves.TryGetValue(name, out var curves))
            {
                return curves.Evaluate(time);
            }
            else
            {
                // Zero Identity Transform (pos 0, rot 0)
                return TransformData.ZeroIdentity();
            }
        }

        private IEnumerable<AnimationCurve> GetAllAnimationCurves()
        {
            yield return handTrackedCurveLeft;
            yield return handTrackedCurveRight;
            yield return handPinchCurveLeft;
            yield return handPinchCurveRight;

            foreach (var curves in handJointCurvesLeft.Values)
            {
                yield return curves.PositionX;
                yield return curves.PositionY;
                yield return curves.PositionZ;
                yield return curves.RotationX;
                yield return curves.RotationY;
                yield return curves.RotationZ;
                yield return curves.RotationW;
            }

            foreach (var curves in handJointCurvesRight.Values)
            {
                yield return curves.PositionX;
                yield return curves.PositionY;
                yield return curves.PositionZ;
                yield return curves.RotationX;
                yield return curves.RotationY;
                yield return curves.RotationZ;
                yield return curves.RotationW;
            }

            foreach(var curves in objectCurves.Values)
            {
                yield return curves.PositionX;
                yield return curves.PositionY;
                yield return curves.PositionZ;
                yield return curves.RotationX;
                yield return curves.RotationY;
                yield return curves.RotationZ;
            }

            yield return cameraCurves.PositionX;
            yield return cameraCurves.PositionY;
            yield return cameraCurves.PositionZ;
            yield return cameraCurves.RotationX;
            yield return cameraCurves.RotationY;
            yield return cameraCurves.RotationZ;
            yield return cameraCurves.RotationW;
            yield return gazeCurves.PositionX;
            yield return gazeCurves.PositionY;
            yield return gazeCurves.PositionZ;
            yield return gazeCurves.RotationX;
            yield return gazeCurves.RotationY;
            yield return gazeCurves.RotationZ;
            yield return gazeCurves.RotationW;
        }

        /// <summary>
        /// Utility function that creates a non-interpolated keyframe suitable for boolean values.
        /// </summary>
        private static void AddBoolKey(AnimationCurve curve, float time, bool value)
        {
            float fvalue = value ? 1.0f : 0.0f;
            // Set tangents and weights such than the input value is cut off and out tangent is constant.
            var keyframe = new Keyframe(time, fvalue, 0.0f, 0.0f, 0.0f, BoolOutWeight);

            keyframe.weightedMode = WeightedMode.Both;
            curve.AddKey(keyframe);
        }

        /// <summary>
        /// Add a float value to an animation curve.
        /// </summary>
        private static void AddFloatKey(AnimationCurve curve, float time, float value)
        {
            // Use linear interpolation by setting tangents and weights to zero.
            var keyframe = new Keyframe(time, value, 0.0f, 0.0f, 0.0f, 0.0f);

            keyframe.weightedMode = WeightedMode.Both;
            curve.AddKey(keyframe);
        }

        /// <summary>
        /// Add a vector value to an animation curve.
        /// </summary>
        private static void AddVectorKey(AnimationCurve curveX, AnimationCurve curveY, AnimationCurve curveZ, float time, Vector3 vector)
        {
            curveX.AddKey(time, vector.x);
            curveY.AddKey(time, vector.y);
            curveZ.AddKey(time, vector.z);
        }

        /// <summary>
        /// Add a pose keyframe to an animation curve.
        /// Keys are only added if the value changes sufficiently.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        private static void AddPoseKeyFiltered(PoseCurves curves, float time, MixedRealityPose pose, float positionThreshold, float rotationThreshold)
        {
            AddPositionKeyFiltered(curves.PositionX, curves.PositionY, curves.PositionZ, time, pose.Position, positionThreshold);
            AddRotationKeyFiltered(curves.RotationX, curves.RotationY, curves.RotationZ, curves.RotationW, time, pose.Rotation, rotationThreshold);
        }

        /// <summary>
        /// Add a vector keyframe to animation curve if the threshold distance to the previous value is exceeded.
        /// Otherwise replace the last keyframe instead of adding a new one.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        private static void AddPositionKeyFiltered(AnimationCurve curveX, AnimationCurve curveY, AnimationCurve curveZ, float time, Vector3 position, float threshold)
        {
            float sqrThreshold = threshold * threshold;

            int iX = FindKeyframeInterval(curveX, time);
            int iY = FindKeyframeInterval(curveY, time);
            int iZ = FindKeyframeInterval(curveZ, time);

            if (iX > 0 && iY > 0 && iZ > 0)
            {
                var v0 = new Vector3(curveX.keys[iX - 1].value, curveY.keys[iY - 1].value, curveZ.keys[iZ - 1].value);
                var v1 = new Vector3(curveX.keys[iX].value, curveY.keys[iY].value, curveZ.keys[iZ].value);

                // Merge the preceding two intervals if difference is small enough
                if ((v1 - v0).sqrMagnitude <= sqrThreshold && (position - v1).sqrMagnitude <= sqrThreshold)
                {
                    curveX.RemoveKey(iX);
                    curveY.RemoveKey(iY);
                    curveZ.RemoveKey(iZ);
                }
            }

            AddFloatKey(curveX, time, position.x);
            AddFloatKey(curveY, time, position.y);
            AddFloatKey(curveZ, time, position.z);
        }

        /// <summary>
        /// Add a quaternion keyframe to animation curve if the threshold angular difference (in degrees) to the previous value is exceeded.
        /// Otherwise replace the last keyframe instead of adding a new one.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        private static void AddRotationKeyFiltered(AnimationCurve curveX, AnimationCurve curveY, AnimationCurve curveZ, AnimationCurve curveW, float time, Quaternion rotation, float threshold)
        {
            // Precompute the dot product threshold so that dot product can be used for comparison instead of angular difference
            float compThreshold = Mathf.Sqrt((Mathf.Cos(threshold * Mathf.PI / 180f) + 1f) / 2f);
            int iX = FindKeyframeInterval(curveX, time);
            int iY = FindKeyframeInterval(curveY, time);
            int iZ = FindKeyframeInterval(curveZ, time);
            int iW = FindKeyframeInterval(curveW, time);

            if (iX > 0 && iY > 0 && iZ > 0 && iW > 0)
            {
                var v0 = new Quaternion(curveX.keys[iX - 1].value, curveY.keys[iY - 1].value, curveZ.keys[iZ - 1].value, curveW.keys[iW - 1].value);
                var v1 = new Quaternion(curveX.keys[iX].value, curveY.keys[iY].value, curveZ.keys[iZ].value, curveW.keys[iW].value);

                // Merge the preceding two intervals if difference is small enough
                if (Quaternion.Dot(v0, v1) >= compThreshold && Quaternion.Dot(rotation, v1) >= compThreshold)
                {
                    curveX.RemoveKey(iX);
                    curveY.RemoveKey(iY);
                    curveZ.RemoveKey(iZ);
                    curveW.RemoveKey(iW);
                }
            }

            AddFloatKey(curveX, time, rotation.x);
            AddFloatKey(curveY, time, rotation.y);
            AddFloatKey(curveZ, time, rotation.z);
            AddFloatKey(curveW, time, rotation.w);
        }

        private static void PoseCurvesToStream(BinaryWriter writer, PoseCurves curves)
        {
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.PositionX);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.PositionY);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.PositionZ);

            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.RotationX);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.RotationY);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.RotationZ);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.RotationW);
            
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.ScaleX);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.ScaleY);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.ScaleZ);
        }

        private static void PoseCurvesFromStream(BinaryReader reader, PoseCurves curves)
        {
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.PositionX);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.PositionY);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.PositionZ);

            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.RotationX);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.RotationY);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.RotationZ);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.RotationW);

            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.ScaleX);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.ScaleY);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.ScaleZ);
        }

        private static void RayCurvesToStream(BinaryWriter writer, RayCurves curves)
        {
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.OriginX);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.OriginY);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.OriginZ);

            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.DirectionX);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.DirectionY);
            InputAnimationSerializationUtils.WriteFloatCurveSimple(writer, curves.DirectionZ);
        }

        private static void RayCurvesFromStream(BinaryReader reader, RayCurves curves)
        {
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.OriginX);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.OriginY);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.OriginZ);

            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.DirectionX);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.DirectionY);
            InputAnimationSerializationUtils.ReadFloatCurveSimple(reader, curves.DirectionZ);
        }

        private static void ObjectCurvesToStream(BinaryWriter writer, Dictionary<string, PoseCurves> objectCurves)
        {
            writer.Write(objectCurves.Count);
            foreach(var entry in objectCurves)
            {
                writer.Write(entry.Key);
                PoseCurvesToStream(writer, entry.Value);
            }
        }

        private static void ObjectCurvesFromStream(BinaryReader reader, Dictionary<string, PoseCurves> objectCurves)
        {
            int objectCount = reader.ReadInt32();
            for (int i = 0; i < objectCount; i++)
            {
                string name = reader.ReadString();
                if (!objectCurves.TryGetValue(name, out var curves))
                {
                    curves = new PoseCurves();
                    objectCurves.Add(name, curves);
                }
                PoseCurvesFromStream(reader, curves);
            }
        }

        /// <summary>
        /// Removes points from a set of curves representing a 3D position, such that the error resulting from removing a point never exceeds 'threshold' units.
        /// </summary>
        /// <param name="threshold">The maximum permitted error between the old and new curves, in units.</param>
        /// <param name="partitionSize">The size of the partitions of the curves that will be optimized independently. Larger values will optimize the curves better, but may take longer.</param>
        /// <remarks>Uses the Ramer–Douglas–Peucker algorithm</remarks>
        private static void OptimizePositionCurve(ref AnimationCurve curveX, ref AnimationCurve curveY, ref AnimationCurve curveZ, float threshold, int partitionSize)
        {
            float sqrThreshold = threshold * threshold;
            var inCurveX = curveX;
            var inCurveY = curveY;
            var inCurveZ = curveZ;
            // Create new curves to avoid deleting points while iterating.
            var outCurveX = new AnimationCurve();
            var outCurveY = new AnimationCurve();
            var outCurveZ = new AnimationCurve();

            outCurveX.AddKey(curveX[0]);
            outCurveY.AddKey(curveY[0]);
            outCurveZ.AddKey(curveZ[0]);

            if (partitionSize == 0)
            {
                Recurse(0, curveX.length - 1);
                outCurveX.AddKey(curveX[curveX.length - 1]);
                outCurveY.AddKey(curveY[curveY.length - 1]);
                outCurveZ.AddKey(curveZ[curveZ.length - 1]);
            }
            else
            {
                for (int i = 0, j = partitionSize; i < curveX.length - partitionSize; i += partitionSize, j = Mathf.Min(j + partitionSize, curveX.length - 1))
                {
                    Recurse(i, j);
                    outCurveX.AddKey(curveX[j]);
                    outCurveY.AddKey(curveY[j]);
                    outCurveZ.AddKey(curveZ[j]);
                }
            }

            curveX = outCurveX;
            curveY = outCurveY;
            curveZ = outCurveZ;

            void Recurse(int start, int end)
            {
                if (start + 1 >= end - 1)
                {
                    return;
                }

                int bestIndex = -1;
                float bestDistance = 0f;
                float startTime = inCurveX[start].time;
                float endTime = inCurveX[end].time;
                var startPosition = new Vector3(inCurveX[start].value, inCurveY[start].value, inCurveZ[start].value);
                var endPosition = new Vector3(inCurveX[end].value, inCurveY[end].value, inCurveZ[end].value);

                for (int i = start + 1; i <= end - 1; i++)
                {
                    var position = new Vector3(inCurveX[i].value, inCurveY[i].value, inCurveZ[i].value);
                    var interp = Vector3.Lerp(startPosition, endPosition, Mathf.InverseLerp(startTime, endTime, inCurveX[i].time));

                    float distance = (position - interp).sqrMagnitude;

                    if (distance > bestDistance)
                    {
                        bestIndex = i;
                        bestDistance = distance;
                    }
                }

                if (bestDistance < sqrThreshold || bestIndex < 0)
                {
                    return;
                }

                outCurveX.AddKey(inCurveX[bestIndex]);
                outCurveY.AddKey(inCurveY[bestIndex]);
                outCurveZ.AddKey(inCurveZ[bestIndex]);
                Recurse(start, bestIndex);
                Recurse(bestIndex, end);
            }
        }

        /// <summary>
        /// Removes points from a set of curves representing a 3D direction vector, such that the error resulting from removing a point never exceeds 'threshold' degrees.
        /// </summary>
        /// <param name="threshold">The maximum permitted error between the old and new curves, in degrees.</param>
        /// <param name="partitionSize">The size of the partitions of the curves that will be optimized independently. Larger values will optimize the curves better, but may take longer.</param>
        /// <remarks>Uses the Ramer–Douglas–Peucker algorithm</remarks>
        private static void OptimizeDirectionCurve(ref AnimationCurve curveX, ref AnimationCurve curveY, ref AnimationCurve curveZ, float threshold, int partitionSize)
        {
            float cosThreshold = Mathf.Cos(threshold * Mathf.PI / 180f);
            var inCurveX = curveX;
            var inCurveY = curveY;
            var inCurveZ = curveZ;
            // Create new curves to avoid deleting points while iterating.
            var outCurveX = new AnimationCurve();
            var outCurveY = new AnimationCurve();
            var outCurveZ = new AnimationCurve();

            outCurveX.AddKey(curveX[0]);
            outCurveY.AddKey(curveY[0]);
            outCurveZ.AddKey(curveZ[0]);

            if (partitionSize == 0)
            {
                Recurse(0, curveX.length - 1);
                outCurveX.AddKey(curveX[curveX.length - 1]);
                outCurveY.AddKey(curveY[curveY.length - 1]);
                outCurveZ.AddKey(curveZ[curveZ.length - 1]);
            }
            else
            {
                for (int i = 0, j = partitionSize; i < curveX.length - partitionSize; i += partitionSize, j = Mathf.Min(j + partitionSize, curveX.length - 1))
                {
                    Recurse(i, j);
                    outCurveX.AddKey(curveX[j]);
                    outCurveY.AddKey(curveY[j]);
                    outCurveZ.AddKey(curveZ[j]);
                }
            }

            curveX = outCurveX;
            curveY = outCurveY;
            curveZ = outCurveZ;

            void Recurse(int start, int end)
            {
                if (start + 1 >= end - 1)
                {
                    return;
                }

                int bestIndex = -1;
                float bestDot = 1f;
                float startTime = inCurveX[start].time;
                float endTime = inCurveX[end].time;
                var startPosition = new Vector3(inCurveX[start].value, inCurveY[start].value, inCurveZ[start].value);
                var endPosition = new Vector3(inCurveX[end].value, inCurveY[end].value, inCurveZ[end].value);

                for (int i = start + 1; i <= end - 1; i++)
                {
                    var position = new Vector3(inCurveX[i].value, inCurveY[i].value, inCurveZ[i].value);
                    var interp = Vector3.Lerp(startPosition, endPosition, Mathf.InverseLerp(startTime, endTime, inCurveX[i].time)).normalized;

                    float dot = Vector3.Dot(position, interp);

                    if (dot < bestDot)
                    {
                        bestIndex = i;
                        bestDot = dot;
                    }
                }

                if (bestDot > cosThreshold || bestIndex < 0)
                {
                    return;
                }

                outCurveX.AddKey(inCurveX[bestIndex]);
                outCurveY.AddKey(inCurveY[bestIndex]);
                outCurveZ.AddKey(inCurveZ[bestIndex]);
                Recurse(start, bestIndex);
                Recurse(bestIndex, end);
            }
        }

        /// <summary>
        /// Removes points from a set of curves representing a quaternion, such that the error resulting from removing a point never exceeds 'threshold' degrees.
        /// </summary>
        /// <param name="threshold">The maximum permitted error between the old and new curves, in degrees</param>
        /// <param name="partitionSize">The size of the partitions of the curves that will be optimized independently. Larger values will optimize the curves better, but may take longer.</param>
        /// <remarks>Uses the Ramer–Douglas–Peucker algorithm</remarks>
        private static void OptimizeRotationCurve(ref AnimationCurve curveX, ref AnimationCurve curveY, ref AnimationCurve curveZ, ref AnimationCurve curveW, float threshold, int partitionSize)
        {
            float compThreshold = Mathf.Sqrt((Mathf.Cos(threshold * Mathf.PI / 180f) + 1f) / 2f);
            var inCurveX = curveX;
            var inCurveY = curveY;
            var inCurveZ = curveZ;
            var inCurveW = curveW;
            // Create new curves to avoid deleting points while iterating.
            var outCurveX = new AnimationCurve();
            var outCurveY = new AnimationCurve();
            var outCurveZ = new AnimationCurve();
            var outCurveW = new AnimationCurve();

            outCurveX.AddKey(curveX[0]);
            outCurveY.AddKey(curveY[0]);
            outCurveZ.AddKey(curveZ[0]);
            outCurveW.AddKey(curveW[0]);

            if (partitionSize == 0)
            {
                Recurse(0, curveX.length - 1);
                outCurveX.AddKey(curveX[curveX.length - 1]);
                outCurveY.AddKey(curveY[curveY.length - 1]);
                outCurveZ.AddKey(curveZ[curveZ.length - 1]);
                outCurveW.AddKey(curveZ[curveW.length - 1]);
            }
            else
            {
                for (int i = 0, j = partitionSize; i < curveX.length - partitionSize; i += partitionSize, j = Mathf.Min(j + partitionSize, curveX.length - 1))
                {
                    Recurse(i, j);
                    outCurveX.AddKey(curveX[j]);
                    outCurveY.AddKey(curveY[j]);
                    outCurveZ.AddKey(curveZ[j]);
                    outCurveZ.AddKey(curveW[j]);
                }
            }

            curveX = outCurveX;
            curveY = outCurveY;
            curveZ = outCurveZ;
            curveW = outCurveW;

            void Recurse(int start, int end)
            {
                if (start + 1 >= end - 1)
                {
                    return;
                }

                int bestIndex = -1;
                float bestDot = 1f;
                float startTime = inCurveX[start].time;
                float endTime = inCurveX[end].time;
                var startRotation = new Quaternion(inCurveX[start].value, inCurveY[start].value, inCurveZ[start].value, inCurveW[start].value).normalized;
                var endRotation = new Quaternion(inCurveX[end].value, inCurveY[end].value, inCurveZ[end].value, inCurveW[end].value).normalized;

                for (int i = start + 1; i <= end - 1; i++)
                {
                    var rotation = new Quaternion(inCurveX[i].value, inCurveY[i].value, inCurveZ[i].value, inCurveW[i].value).normalized;
                    var interp = Quaternion.Lerp(startRotation, endRotation, Mathf.InverseLerp(startTime, endTime, inCurveX[i].time));

                    float dot = Quaternion.Dot(rotation, interp);

                    if (dot < bestDot)
                    {
                        bestIndex = i;
                        bestDot = dot;
                    }
                }

                if (bestDot > compThreshold || bestIndex < 0)
                {
                    return;
                }

                outCurveX.AddKey(inCurveX[bestIndex]);
                outCurveY.AddKey(inCurveY[bestIndex]);
                outCurveZ.AddKey(inCurveZ[bestIndex]);
                Recurse(start, bestIndex);
                Recurse(bestIndex, end);
            }
        }

        /// <summary>
        /// Utility function that creates a non-interpolated keyframe suitable for boolean values.
        /// Keys are only added if the value changes.
        /// Returns the index of the newly added keyframe, or -1 if no keyframe has been added.
        /// </summary>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        private static int AddBoolKeyFiltered(AnimationCurve curve, float time, bool value)
        {
            float fvalue = value ? 1.0f : 0.0f;
            // Set tangents and weights such than the input value is cut off and out tangent is constant.
            var keyframe = new Keyframe(time, fvalue, 0.0f, 0.0f, 0.0f, BoolOutWeight);
            keyframe.weightedMode = WeightedMode.Both;

            int insertAfter = FindKeyframeInterval(curve, time);
            if (insertAfter >= 0 && curve.keys[insertAfter].value == fvalue)
            {
                // Value unchanged from previous key, ignore
                return -1;
            }

            int insertBefore = insertAfter + 1;
            if (insertBefore < curve.keys.Length && curve.keys[insertBefore].value == fvalue)
            {
                // Same value as next key, replace next key
                return curve.MoveKey(insertBefore, keyframe);
            }

            return curve.AddKey(keyframe);
        }

        /// <summary>
        /// Find an index i in the sorted events list, such that events[i].time &lt;= time &lt; events[i+1].time.
        /// </summary>
        /// <returns>
        /// 0 &lt;= i &lt; eventCount if a full interval could be found.
        /// -1 if time is less than the first event time.
        /// eventCount-1 if time is greater than the last event time.
        /// </returns>
        /// <remarks>
        /// Uses binary search.
        /// </remarks>
        [Obsolete("Use FromRecordingBuffer to construct new InputAnimations")]
        private static int FindKeyframeInterval(AnimationCurve curve, float time)
        {
            var keys = curve.keys;
            int lowIdx = -1;
            int highIdx = keys.Length;
            while (lowIdx < highIdx - 1)
            {
                int midIdx = (lowIdx + highIdx) >> 1;
                if (time >= keys[midIdx].time)
                {
                    lowIdx = midIdx;
                }
                else
                {
                    highIdx = midIdx;
                }
            }
            return lowIdx;
        }
    }

}