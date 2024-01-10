# 4D Holographic Tutorials

### Abstract
Learning tasks that involve hand-object manipulation using traditional
methods, such as learning from written manuals or YouTube tutorials,
can be challenging due to the importance of spatial information. Aug-
mented Reality applications like Microsoft Dynamics 365 Guides have
become increasingly popular because they provide information in 3D,
which can reduce errors and learning time. However, creating step-by-
step instructions in 3D can be time-consuming, especially when experts
need to interact with objects using tools. Some recent approaches have
attempted to address this challenge by capturing hand poses during
authoring sessions. However, this method is not scalable as it cannot
capture manipulated objects that are not trackable. To address this
issue, we propose capturing 3D tutorial videos in the form of 3D point
clouds with the 3D pose relative to the target object using mixed reality
devices like HoloLens 2. These point clouds can then be shown to
users as 3D animations. To achieve our proposed method of capturing
3D tutorial videos, we utilize Azure Object Anchor to estimate the 6D
pose of the target object. We also employ segmentation-based methods
to detect and remove irrelevant information, such as the background.
This combination of techniques allows for more accurate and effective
capture of 3D point clouds with the necessary spatial information

Bachelor Thesis, Spring 2023  

Yanik Künzi


### Usage
To play back a tutorial, make sure to do follow the following steps:
* Make sure the 3D model of the objects you are working with are present in the `3D Objects` directory on the HoloLens
* As soon as an object is recognized, a new window appears allowing the user to select and play back tutorial
