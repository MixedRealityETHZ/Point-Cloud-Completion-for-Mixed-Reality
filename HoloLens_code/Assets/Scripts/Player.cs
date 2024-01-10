using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit.UI;
using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;



namespace Tutorials
{

    /// <summary>
    /// Monobehaviour class that listens to the FileHandler's Animation List instance and plays any animation from it as soon as the current animation changes. 
    /// </summary>
    /// <seealso cref="UnityEngine.MonoBehaviour" />
    public class Player : MonoBehaviour
    {
        private static readonly int jointCount = Enum.GetNames(typeof(TrackedHandJoint)).Length;


        private bool isPlaying = false;

        /// <summary>
        /// Duration of the played animation.
        /// </summary>
        private float Duration;

        private float localTime = 0.0f;

        /// <summary>
        /// The actual animation that is being replayed 
        /// </summary>
        private new InputAnimation animation;

        [SerializeField]
        private Transform animationSpecificPointOfReference;

        [SerializeField]
        [Tooltip("The hand model that will be animated using the animation data of the currently loaded animation. This must match the structure used by RiggedHandVisualizer.")]
        private GameObject riggedHandLeftPrefab;


        /// <summary>
        /// The hand object in the scene, instantiated from riggedHandLeftPrefab at runtime. This will be animated. 
        /// </summary>
        private GameObject riggedHandLeft;

        /// <summary>
        /// The rigged hand visualizer that evaluates the hand pose in each frame
        /// </summary>
        private RiggedHandVisualizer riggedHandVisualizerLeft;

        [SerializeField]
        [Tooltip("The hand model that will be animated using the animation data of the currently loaded animation. This must match the structure used by RiggedHandVisualizer.")]
        private GameObject riggedHandRightPrefab;

        /// <summary>
        /// The hand object in the scene, instantiated from riggedHandLeftPrefab at runtime. This will be animated. 
        /// </summary>
        private GameObject riggedHandRight;

        /// <summary>
        /// The rigged hand visualizer that evaluates the hand pose in each frame
        /// </summary>
        private RiggedHandVisualizer riggedHandVisualizerRight;

        [SerializeField]
        private ObjectManager objectManager;
        /// <summary>
        /// Is set at the start of each animation to the objects to be animated
        /// </summary>
        private List<GameObject> objectList;

        /// <summary>
        /// A list of essential components to retain
        /// </summary>
        public static readonly string[] essentialComponents = { "Transform", "MeshFilter", "MeshRenderer", "ConstraintManager" };

        /// <summary>
        /// Material to be used for the animated objects
        /// </summary>
        [SerializeField]
        private Material animatingMaterial;

        private bool startAgain = false;

        public void Start()
        {
            // instantiating the left hand object that will be animated
            riggedHandLeft = Instantiate(riggedHandLeftPrefab, animationSpecificPointOfReference.position, animationSpecificPointOfReference.rotation);

            riggedHandLeft.SetActive(false);
            riggedHandLeft.transform.parent = animationSpecificPointOfReference;
            riggedHandLeft.name = "PlaybackServiceLeftHand";

            riggedHandVisualizerLeft = riggedHandLeft.GetComponent<RiggedHandVisualizer>();

            // instantiating the right hand object that will be animated
            riggedHandRight = Instantiate(riggedHandRightPrefab, animationSpecificPointOfReference.position, animationSpecificPointOfReference.rotation);

            riggedHandRight.SetActive(false);
            riggedHandRight.transform.parent = animationSpecificPointOfReference;
            riggedHandRight.name = "PlaybackServiceRightHand";
            riggedHandVisualizerRight = riggedHandRight.GetComponent<RiggedHandVisualizer>();
        }

        /// <summary>
        /// Sets the animation that is currently being replay to 100%. 
        /// </summary>
        public void StartAgain()
        {
            startAgain = true;
        }

        /// <summary>
        /// Plays the currently loaded animation from beginning. If no animation is loaded, the animation specific point of reference is reset and the replay is stopped. 
        /// </summary>
        public void Play()
        {
        }

        /// <summary>
        /// Remove all components not needed for animation
        /// </summary>
        private void AdjustComponents(GameObject obj)
        {
            foreach (var comp in obj.GetComponents<Component>())
            {
                if (!essentialComponents.Contains(comp.GetType().Name))
                {
                    Destroy(comp);
                }
            }
            Destroy(obj.GetComponent<ConstraintManager>());
            obj.GetComponent<MeshRenderer>().material = animatingMaterial;
        }

        /// <summary>
        /// Create clones of every object used in the animation and add to list.
        /// Update the curves associated name
        /// </summary>
        public void InstantiateObjects()
        {
            // No recording exists at the moment
            if (animation == null) return;

            // Initialize a new object list
            objectList = new List<GameObject>();


            foreach(var entry in animation.objectCurves)
            {
                GameObject obj = null;
                int index = entry.Key.IndexOf("-");
                if (index < 0)
                {
                    Debug.Log("entry is " + entry.Key);
                    if (entry.Key == "userdefined(Clone)")
                    {
                        // Special case of user created object
                        obj = objectManager.GetUserDefinedObject();
                        if (obj == null)
                        {
                            Debug.Log("Cannot locate the user-defined object");
                        }
                    }
                    else
                    {
                        Debug.Log($"Cannot map {entry.Key} to original objects. Make sure to use the correct naming convention for objects.");
                    }
                }
                else
                {
                    string originalName = entry.Key.Substring(0, index).Trim();
                    obj = objectManager.GetOriginalObject(originalName);
                    if (obj == null)
                    {
                        Debug.Log($"GameObject with name {originalName} could not be found.");
                        continue;
                    }
                }

                if (obj != null)
                {
                    GameObject clone = Instantiate(obj, animationSpecificPointOfReference);
                    AdjustComponents(clone);
                    clone.SetActive(false);
                    clone.name = entry.Key;
                    objectList.Add(clone);
                }
                
            }
        }

        /// <summary>
        /// Stops the playback
        /// </summary>
        public void Stop()
        {
            if (!isPlaying)
            {
                return;
            }
            objectManager.ActivateRealObjects();
            localTime = 0.0f;
            isPlaying = false;
            Evaluate();
        }

        public void Update()
        {
        }

        /// Evaluate the animation at localTime
        private void Evaluate()
        {
            // if no replay is currently happening, both hands and all objects are set invisible
            if (!isPlaying)
            {
                riggedHandLeft.gameObject.SetActive(false);
                riggedHandRight.gameObject.SetActive(false);
                foreach(var obj in objectList)
                {
                    obj.SetActive(false);
                }
                return;
            }
            // otherwise, only the hands that are supposed to be visible are set to visible state. 
            else
            {
            }

            if (animation == null)
            {
                localTime = 0.0f;
                isPlaying = false;

                return;
            }

            // evaluate hand poses
            if (animation.HasHandData)
            {
            }

            EvaluateObjectData();
        }

        /// <summary>
        /// Update object positions and rotations
        /// </summary>
        private void EvaluateObjectData()
        {
            foreach(var obj in objectList)
            {
                TransformData data = animation.EvaluateObject(localTime, obj.name);
                obj.transform.localPosition =data.GetPosition();
                obj.transform.localRotation = data.GetRotation();
                obj.transform.localScale = data.GetScale();
            } 
        }

        /// <summary>
        /// Returns the state of the player
        /// </summary>
        /// <returns>bool</returns>
        public bool IsPlaying()
        {
            return isPlaying;
        }


        /// <summary>
        /// updates the joint positions and rotations in the rigged hand visualizer
        /// </summary>
        /// <param name="handedness">The handedness.</param>
        private void EvaluateHandData(Handedness handedness)
        {
            animation.EvaluateHandState(localTime, handedness, out bool isTracked, out bool isPinching);

            if (handedness == Handedness.Left)
            {
                riggedHandLeft.gameObject.SetActive(isTracked);
            }
            else
            {
                riggedHandRight.gameObject.SetActive(isTracked);
            }

            IDictionary<TrackedHandJoint, TransformData> joints = new Dictionary<TrackedHandJoint, TransformData>();


            for (int i = 0; i < jointCount; ++i)
            {
                joints.Add(new KeyValuePair<TrackedHandJoint, TransformData>(
                    (TrackedHandJoint)i,
                    animation.EvaluateHandJoint(localTime, handedness, (TrackedHandJoint)i)));
            }

            if (handedness == Handedness.Left)
            {
                riggedHandVisualizerLeft.UpdateHandJoints(joints, isTracked);
            }
            if (handedness == Handedness.Right)
            {
                riggedHandVisualizerRight.UpdateHandJoints(joints, isTracked);
            }

        }
    }

}