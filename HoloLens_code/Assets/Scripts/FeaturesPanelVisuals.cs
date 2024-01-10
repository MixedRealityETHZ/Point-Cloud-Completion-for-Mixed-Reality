// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using UnityEngine;

namespace Microsoft.MixedReality.Toolkit.UI
{
    /// <summary>
    /// Class that initializes the appearance of the features panel according to the toggled states of the associated features
    /// </summary>
    internal class FeaturesPanelVisuals : MonoBehaviour
    {
        [SerializeField]
        private Interactable handMeshButton = null;

        private void Start()
        {
            if (CoreServices.InputSystem?.InputSystemProfile != null)
            {
                MixedRealityHandTrackingProfile handProfile = CoreServices.InputSystem.InputSystemProfile.HandTrackingProfile;
                if (handProfile != null)
                {
                    handProfile.EnableHandMeshVisualization = false;
                }
                Debug.Log($"In {name}");
            }
        }
    }
}
