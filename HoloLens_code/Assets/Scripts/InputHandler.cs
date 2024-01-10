using UnityEngine;
using UnityEngine.UI;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using Microsoft.MixedReality.Toolkit.UI;
using Microsoft.MixedReality.Toolkit.Input;
using TMPro;

namespace Tutorials
{
    public class InputHandler : MonoBehaviour
    {
        private string countdownText = "";
        private string shouldStartRecord = "no"; // Using strings because locks don't work on bools
        private CancellationTokenSource cancelRecordingCountdownToken;

        public void Update()
        {
        }

        private void Start()
        {
        }

        /// <summary>
        /// Function to handle speech recording
        /// </summary>
        public void SpeechRecord()
        {
        }

        /// <summary>
        /// Function to handle speech save recording
        /// </summary>
        public void SpeechSave()
        {
        }

        /// <summary>
        /// Record UI button pressed; take correct action to start/stop
        /// </summary>
        public void RecordAction()
        {
        }

        public void StartCountdownThenRecord(CancellationTokenSource ct)
        {
        }

        /// <summary>
        /// Update the description of the current name through user input
        /// </summary>
        public void EditStepName()
        {
        }

        /// <summary>
        /// Updates the step number and step name displayed in the recording panel
        /// </summary>
        private void OnAnimationChanged()
        {
        }

        /// <summary>
        /// Called when recording starts.
        /// </summary>
        private void OnStartRecording()
        {
        }

        /// <summary>
        /// Resets the current animation to the start frame (100% in the loading bar).
        /// </summary>
        public void StartAgain()
        {
        }

        /// <summary>
        /// Function to handle speech play playback
        /// </summary>
        public void SpeechPlay()
        {
        }

        /// <summary>
        /// Function to handle speech stop playback
        /// </summary>
        public void SpeechStop()
        {
        }

        /// <summary>
        /// Play/Stop the current animation when issued
        /// </summary>
        public void PlayAction()
        {
        }
    }
}