// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System.Collections.Generic;
using UnityEngine;


namespace Tutorials
{

    /// <summary>
    /// Hand visualizer that controls a hierarchy of transforms to be used by a SkinnedMeshRenderer
    /// Implementation is derived from LeapMotion RiggedHand and RiggedFinger and has visual parity
    /// 
    /// Note: This class is a component in each respective hand model. It offers control over the movement and animation of the hand models
    /// </summary>
    public class RiggedHandVisualizer : MonoBehaviour
    {
        public GameObject GameObjectProxy => gameObject;

        public IMixedRealityController Controller { get; set; }

        /// <summary>
        /// Wrist Transform
        /// </summary>
        public Transform Wrist;
        /// <summary>
        /// Palm transform
        /// </summary>
        public Transform Palm;
        /// <summary>
        /// Thumb metacarpal transform  (thumb root)
        /// </summary>
        public Transform ThumbRoot;

        /// <summary>
        /// Prefab of a regular joint 
        /// </summary>
        public GameObject JointPrefab;
        /// <summary>
        /// Prefab of the palm joint 
        /// </summary>
        public GameObject PalmJointPrefab;
        /// <summary>
        /// Prefab of a finger tip joint 
        /// </summary>
        public GameObject FingerTipPrefab;

        [Tooltip("First finger node is metacarpal joint.")]
        public bool ThumbRootIsMetacarpal = true;

        /// <summary>
        /// Index metacarpal transform (index finger root)
        /// </summary>
        public Transform IndexRoot;

        [Tooltip("First finger node is metacarpal joint.")]
        public bool IndexRootIsMetacarpal = true;

        /// <summary>
        /// Middle metacarpal transform (middle finger root)
        /// </summary>
        public Transform MiddleRoot;

        [Tooltip("First finger node is metacarpal joint.")]
        public bool MiddleRootIsMetacarpal = true;

        /// <summary>
        /// Ring metacarpal transform (ring finger root)
        /// </summary>
        public Transform RingRoot;

        [Tooltip("Ring finger node is metacarpal joint.")]
        public bool RingRootIsMetacarpal = true;

        /// <summary>
        /// Pinky metacarpal transform (pinky finger root)
        /// </summary>
        public Transform PinkyRoot;

        [Tooltip("First finger node is metacarpal joint.")]
        public bool PinkyRootIsMetacarpal = true;

        [Tooltip("Hands are typically rigged in 3D packages with the palm transform near the wrist. Uncheck this if your model's palm transform is at the center of the palm similar to Leap API hands.")]
        public bool ModelPalmAtLeapWrist = true;

        [Tooltip("Allows the mesh to be stretched to align with finger joint positions. Only set to true when mesh is not visible as this will deform the hand model grotesquely.")]
        public bool DeformPosition = true;

        [Tooltip("Because bones only exist at their roots in model rigs, the length " +
            "of the last fingertip bone is lost when placing bones at positions in the " +
            "tracked hand. " +
            "This option scales the last bone along its X axis (length axis) to match " +
            "its bone length to the tracked bone length. This option only has an " +
            "effect if Deform Positions In Fingers is enabled.")]
        public bool ScaleLastFingerBone = true;

        [Tooltip("If non-zero, this vector and the modelPalmFacing vector " +
        "will be used to re-orient the Transform bones in the hand rig, to " +
        "compensate for bone axis discrepancies between Leap Bones and model " +
        "bones.")]
        public Vector3 ModelFingerPointing = new Vector3(0, 0, 0);

        [Tooltip("If non-zero, this vector and the modelFingerPointing vector " +
            "will be used to re-orient the Transform bones in the hand rig, to " +
            "compensate for bone axis discrepancies between Leap Bones and model " +
            "bones.")]
        public Vector3 ModelPalmFacing = new Vector3(0, 0, 0);

        [SerializeField]
        [Tooltip("Renderer of the hand mesh")]
        private SkinnedMeshRenderer handRenderer = null;

        /// <summary>
        /// Renderer of the hand mesh.
        /// </summary>
        public SkinnedMeshRenderer HandRenderer => handRenderer;

        [SerializeField]
        [Tooltip("Hand material to use for hand tracking hand mesh.")]
        private Material handMaterial = null;

        /// <summary>
        /// Hand material to use for hand tracking hand mesh.
        /// </summary>
        public Material HandMaterial => handMaterial;

        /// <summary>
        /// Property name for modifying the mesh's appearance based on pinch strength
        /// </summary>
        private const string pinchStrengthMaterialProperty = "_PressIntensity";

        /// <summary>
        /// Property name for modifying the mesh's appearance based on pinch strength
        /// </summary>
        public string PinchStrengthMaterialProperty => pinchStrengthMaterialProperty;

        /// <summary>
        /// Precalculated values for LeapMotion testhand fingertip lengths
        /// </summary>
        private const float thumbFingerTipLength = 0.02167f;
        private const float indexingerTipLength = 0.01582f;
        private const float middleFingerTipLength = 0.0174f;
        private const float ringFingerTipLength = 0.0173f;
        private const float pinkyFingerTipLength = 0.01596f;

        /// <summary>
        /// Precalculated fingertip lengths used for scaling the fingertips of the skinnedmesh
        /// to match with tracked hand fingertip size 
        /// </summary>
        private Dictionary<TrackedHandJoint, float> fingerTipLengths = new Dictionary<TrackedHandJoint, float>()
    {
        {TrackedHandJoint.ThumbTip, thumbFingerTipLength },
        {TrackedHandJoint.IndexTip, indexingerTipLength },
        {TrackedHandJoint.MiddleTip, middleFingerTipLength },
        {TrackedHandJoint.RingTip, ringFingerTipLength },
        {TrackedHandJoint.PinkyTip, pinkyFingerTipLength }
    };

        /// <summary>
        /// Rotation derived from the `modelFingerPointing` and
        /// `modelPalmFacing` vectors in the RiggedHand inspector.
        /// </summary>
        private Quaternion userBoneRotation
        {
            get
            {
                if (ModelFingerPointing == Vector3.zero || ModelPalmFacing == Vector3.zero)
                {
                    return Quaternion.identity;
                }
                return Quaternion.Inverse(Quaternion.LookRotation(ModelFingerPointing, -ModelPalmFacing));
            }
        }

        // stores the transforms of each hand joint and matches it with the respective TrackedHandJoint entry
        private Dictionary<TrackedHandJoint, Transform> joints = new Dictionary<TrackedHandJoint, Transform>();

        // contains transforms of individual skeleton joints (not part of the actual hand model, but will show up at the same position. Used to display single joints in an abstract way, e.g. represented by a cube or sphere)
        private Dictionary<TrackedHandJoint, Transform> skeletonJoints = new Dictionary<TrackedHandJoint, Transform>();

        private bool renderSkeletonJoints = true;


        /// <summary>
        /// Gets or sets a value indicating whether to render skeleton joints. Note: If the mesh is rendered (default), most joints will disappear "inside" the hand.
        /// </summary>
        public bool RenderSkeletonJoints { get => renderSkeletonJoints; set => renderSkeletonJoints = value; }

        private bool renderHandMesh = false;

        /// <summary>
        /// Gets or sets a value indicating whether to render hand mesh. this should be true (default) if the hands are supposed to be visible. Otherwise, it's also possible to just decativate the entire GameObject to make it disappear, which is the approach of the recording service. 
        /// </summary>
        /// <value>
        ///   <c>true</c> if [render hand mesh]; otherwise, <c>false</c>.
        /// </value>
        public bool RenderHandMesh { get => renderHandMesh; set => renderHandMesh = value; }


        private void Start()
        {
            // Initialize joint dictionary with their corresponding joint transforms
            joints[TrackedHandJoint.Wrist] = Wrist;
            joints[TrackedHandJoint.Palm] = Palm;

            // Thumb joints, first node is user assigned, note that there are only 4 joints in the thumb
            if (ThumbRoot)
            {
                if (ThumbRootIsMetacarpal)
                {
                    joints[TrackedHandJoint.ThumbMetacarpalJoint] = ThumbRoot;
                    joints[TrackedHandJoint.ThumbProximalJoint] = RetrieveChild(TrackedHandJoint.ThumbMetacarpalJoint);
                }
                else
                {
                    joints[TrackedHandJoint.ThumbProximalJoint] = ThumbRoot;
                }
                joints[TrackedHandJoint.ThumbDistalJoint] = RetrieveChild(TrackedHandJoint.ThumbProximalJoint);
                joints[TrackedHandJoint.ThumbTip] = RetrieveChild(TrackedHandJoint.ThumbDistalJoint);
            }
            // Look up index finger joints below the index finger root joint
            if (IndexRoot)
            {
                if (IndexRootIsMetacarpal)
                {
                    joints[TrackedHandJoint.IndexMetacarpal] = IndexRoot;
                    joints[TrackedHandJoint.IndexKnuckle] = RetrieveChild(TrackedHandJoint.IndexMetacarpal);
                }
                else
                {
                    joints[TrackedHandJoint.IndexKnuckle] = IndexRoot;
                }
                joints[TrackedHandJoint.IndexMiddleJoint] = RetrieveChild(TrackedHandJoint.IndexKnuckle);
                joints[TrackedHandJoint.IndexDistalJoint] = RetrieveChild(TrackedHandJoint.IndexMiddleJoint);
                joints[TrackedHandJoint.IndexTip] = RetrieveChild(TrackedHandJoint.IndexDistalJoint);
            }

            // Look up middle finger joints below the middle finger root joint
            if (MiddleRoot)
            {
                if (MiddleRootIsMetacarpal)
                {
                    joints[TrackedHandJoint.MiddleMetacarpal] = MiddleRoot;
                    joints[TrackedHandJoint.MiddleKnuckle] = RetrieveChild(TrackedHandJoint.MiddleMetacarpal);
                }
                else
                {
                    joints[TrackedHandJoint.MiddleKnuckle] = MiddleRoot;
                }
                joints[TrackedHandJoint.MiddleMiddleJoint] = RetrieveChild(TrackedHandJoint.MiddleKnuckle);
                joints[TrackedHandJoint.MiddleDistalJoint] = RetrieveChild(TrackedHandJoint.MiddleMiddleJoint);
                joints[TrackedHandJoint.MiddleTip] = RetrieveChild(TrackedHandJoint.MiddleDistalJoint);
            }

            // Look up ring finger joints below the ring finger root joint
            if (RingRoot)
            {
                if (RingRootIsMetacarpal)
                {
                    joints[TrackedHandJoint.RingMetacarpal] = RingRoot;
                    joints[TrackedHandJoint.RingKnuckle] = RetrieveChild(TrackedHandJoint.RingMetacarpal);
                }
                else
                {
                    joints[TrackedHandJoint.RingKnuckle] = RingRoot;
                }
                joints[TrackedHandJoint.RingMiddleJoint] = RetrieveChild(TrackedHandJoint.RingKnuckle);
                joints[TrackedHandJoint.RingDistalJoint] = RetrieveChild(TrackedHandJoint.RingMiddleJoint);
                joints[TrackedHandJoint.RingTip] = RetrieveChild(TrackedHandJoint.RingDistalJoint);
            }

            // Look up pinky joints below the pinky root joint
            if (PinkyRoot)
            {
                if (PinkyRootIsMetacarpal)
                {
                    joints[TrackedHandJoint.PinkyMetacarpal] = PinkyRoot;
                    joints[TrackedHandJoint.PinkyKnuckle] = RetrieveChild(TrackedHandJoint.PinkyMetacarpal);
                }
                else
                {
                    joints[TrackedHandJoint.PinkyKnuckle] = PinkyRoot;
                }
                joints[TrackedHandJoint.PinkyMiddleJoint] = RetrieveChild(TrackedHandJoint.PinkyKnuckle);
                joints[TrackedHandJoint.PinkyDistalJoint] = RetrieveChild(TrackedHandJoint.PinkyMiddleJoint);
                joints[TrackedHandJoint.PinkyTip] = RetrieveChild(TrackedHandJoint.PinkyDistalJoint);
            }

            // Give the hand mesh its own material to avoid modifying both hand materials when making property changes
            var handMaterialInstance = new Material(handMaterial);
            handRenderer.sharedMaterial = handMaterialInstance;
        }

        /// <summary>
        /// Gets the parent tracked hand joint if one exists
        /// </summary>
        /// <param name="handJoint">The child hand joint.</param>
        private TrackedHandJoint GetParentTrackedHandJoint(TrackedHandJoint handJoint)
        {
            switch (handJoint)
            {
                case TrackedHandJoint.Palm:
                    return TrackedHandJoint.None;
                case TrackedHandJoint.Wrist:
                    return TrackedHandJoint.None;

                case TrackedHandJoint.ThumbMetacarpalJoint:
                    return TrackedHandJoint.Wrist;
                case TrackedHandJoint.ThumbProximalJoint:
                    if (ThumbRootIsMetacarpal) return TrackedHandJoint.ThumbMetacarpalJoint;
                    else return TrackedHandJoint.Wrist;
                case TrackedHandJoint.ThumbDistalJoint:
                    return TrackedHandJoint.ThumbProximalJoint;
                case TrackedHandJoint.ThumbTip:
                    return TrackedHandJoint.ThumbDistalJoint;

                case TrackedHandJoint.IndexMetacarpal:
                    return TrackedHandJoint.Wrist;
                case TrackedHandJoint.IndexKnuckle:
                    if (IndexRootIsMetacarpal) return TrackedHandJoint.IndexMetacarpal;
                    else return TrackedHandJoint.Wrist;
                case TrackedHandJoint.IndexMiddleJoint:
                    return TrackedHandJoint.IndexKnuckle;
                case TrackedHandJoint.IndexDistalJoint:
                    return TrackedHandJoint.IndexMiddleJoint;
                case TrackedHandJoint.IndexTip:
                    return TrackedHandJoint.IndexDistalJoint;

                case TrackedHandJoint.MiddleMetacarpal:
                    return TrackedHandJoint.Wrist;
                case TrackedHandJoint.MiddleKnuckle:
                    if (MiddleRootIsMetacarpal) return TrackedHandJoint.MiddleMetacarpal;
                    else return TrackedHandJoint.Wrist;
                case TrackedHandJoint.MiddleMiddleJoint:
                    return TrackedHandJoint.MiddleKnuckle;
                case TrackedHandJoint.MiddleDistalJoint:
                    return TrackedHandJoint.MiddleMiddleJoint;
                case TrackedHandJoint.MiddleTip:
                    return TrackedHandJoint.MiddleDistalJoint;

                case TrackedHandJoint.RingMetacarpal:
                    return TrackedHandJoint.Wrist;
                case TrackedHandJoint.RingKnuckle:
                    if (RingRootIsMetacarpal) return TrackedHandJoint.RingMetacarpal;
                    else return TrackedHandJoint.Wrist;
                case TrackedHandJoint.RingMiddleJoint:
                    return TrackedHandJoint.RingKnuckle;
                case TrackedHandJoint.RingDistalJoint:
                    return TrackedHandJoint.RingMiddleJoint;
                case TrackedHandJoint.RingTip:
                    return TrackedHandJoint.RingDistalJoint;

                case TrackedHandJoint.PinkyMetacarpal:
                    return TrackedHandJoint.Wrist;
                case TrackedHandJoint.PinkyKnuckle:
                    if (PinkyRootIsMetacarpal) return TrackedHandJoint.PinkyMetacarpal;
                    else return TrackedHandJoint.Wrist;
                case TrackedHandJoint.PinkyMiddleJoint:
                    return TrackedHandJoint.PinkyKnuckle;
                case TrackedHandJoint.PinkyDistalJoint:
                    return TrackedHandJoint.PinkyMiddleJoint;
                case TrackedHandJoint.PinkyTip:
                    return TrackedHandJoint.PinkyDistalJoint;
            }

            return TrackedHandJoint.None;
        }

        /// <summary>
        /// Retrieves the child if one exists
        /// </summary>
        /// <param name="parentJoint">The parent joint.</param>
        private Transform RetrieveChild(TrackedHandJoint parentJoint)
        {
            if (joints[parentJoint] != null && joints[parentJoint].childCount > 0)
            {
                return joints[parentJoint].GetChild(0);
            }
            return null;
        }

        /// <summary>
        /// Updates the hand joints with position and rotation information in a given dictionary (given by e.g. the playback service)
        /// </summary>
        /// <param name="jointsTransformData">The joints transform data dictionary.</param>
        /// <param name="isTracked">true if this hand is tracked in this frame</param>
        public void UpdateHandJoints(IDictionary<TrackedHandJoint, TransformData> jointsTransformData, bool isTracked)
        {
            if (renderSkeletonJoints)
            {
                // go through all joints and update the skeleton joint position and rotation (will be the same positions and rotations as the regular hand model joints)
                foreach (TrackedHandJoint handJoint in jointsTransformData.Keys)
                {
                    Transform skeletonJointTransform;

                    // if this transform has been instantiated before, only update the position and location
                    if (skeletonJoints.TryGetValue(handJoint, out skeletonJointTransform))
                    {
                        skeletonJointTransform.localPosition = jointsTransformData[handJoint].GetPosition();
                        skeletonJointTransform.localRotation = jointsTransformData[handJoint].GetRotation();

                        skeletonJointTransform.gameObject.SetActive(isTracked);
                    }
                    // otherwise instantiate a new Prefab/GameObject
                    else
                    {
                        GameObject prefab;
                        if (handJoint == TrackedHandJoint.None || handJoint == TrackedHandJoint.Palm)
                        {
                            // No visible mesh for the "None" joint
                            prefab = null;
                        }
                        else if (handJoint == TrackedHandJoint.Wrist)
                        {
                            prefab = null;// PalmJointPrefab;
                        }
                        else if (handJoint == TrackedHandJoint.IndexTip)
                        {
                            prefab = FingerTipPrefab;
                        }
                        else
                        {
                            prefab = null;// JointPrefab;
                        }

                        GameObject jointObject;
                        if (prefab != null)
                        {
                            jointObject = Instantiate(prefab);
                        }
                        else
                        {
                            jointObject = new GameObject();
                        }

                        jointObject.name = handJoint.ToString() + " Proxy Transform";
                        jointObject.transform.localPosition = jointsTransformData[handJoint].GetPosition();
                        jointObject.transform.localRotation = jointsTransformData[handJoint].GetRotation();

                        Transform parent;
                        skeletonJoints.TryGetValue(GetParentTrackedHandJoint(handJoint), out parent);

                        jointObject.transform.parent = (parent != null) ? parent : transform;

                        jointObject.SetActive(isTracked);

                        skeletonJoints.Add(handJoint, jointObject.transform);
                    }
                }
            }
            else
            {
                // clear existing joint GameObjects / meshes if they shouldn't be displayed
                foreach (var joint in skeletonJoints)
                {
                    Destroy(joint.Value.gameObject);
                }

                skeletonJoints.Clear();
            }

            if (renderHandMesh)
            {
                // Render the rigged hand mesh itself
                Transform jointTransform;
                // Apply updated TrackedHandJoint pose data to the assigned transforms
                foreach (TrackedHandJoint handJoint in jointsTransformData.Keys)
                {
                    if (handJoint == TrackedHandJoint.None
                        || handJoint == TrackedHandJoint.Palm
                        || handJoint == TrackedHandJoint.IndexMetacarpal
                        || handJoint == TrackedHandJoint.MiddleMetacarpal
                        || handJoint == TrackedHandJoint.RingMetacarpal
                        || handJoint == TrackedHandJoint.PinkyMetacarpal
                        )
                    {
                        continue;
                    }

                    if (handJoint == TrackedHandJoint.ThumbMetacarpalJoint)
                    {

                    }

                    if (joints.TryGetValue(handJoint, out jointTransform))
                    {
                        if (jointTransform != null)
                        {
                            jointTransform.localRotation = jointsTransformData[handJoint].GetRotation();

                            // for most joints (all except first, usually palm joint) only the rotation is set. The position relative to the parent joint always remains the same (just like in a real hand, where joints are connected by fixed bones.)
                            if (DeformPosition || handJoint == TrackedHandJoint.Wrist)
                            {
                                jointTransform.localPosition = jointsTransformData[handJoint].GetPosition();
                            }

                        }
                    }
                }

            }
        }
    }

}