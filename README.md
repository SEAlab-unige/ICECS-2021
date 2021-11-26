# An Affordance Detection Pipeline for Resource-Constrained Devices
This repository contains a demo of the pipeline described in the paper **"An Affordance Detection Pipeline for Resource-Constrained Devices"** by Apicella, T. et al.

## Authors 
* Tommaso Apicella
* Andrea Cavallaro  
* Riccardo Berta  
* Paolo Gastaldo 
* Francesco Bellotti
* Edoardo Ragusa

## Table of contents
* [Python demo](#python-demo)
* [Reference](#reference)

## Python demo
### Requirements
The requirements to run the python code are the following:
* Python 3.6
* Tensorflow 2.5
* Numpy
* OpenCV
* Keras segmentation

For additional details, see *requirements.txt* file.

### Description
The `demo_object_detection_affordance.py` runs the pipeline described in the **paper** on a mp4 video.
The object detector SavedModel format is in *object_detector* folder, while affordance detector weights and config files are available in *affordance_detector* folder.
The object detector crops the objects present in the scene and the affordance detector segments the patches pixel-wise in three classes: Background (black), Grasp (blue) and No-grasp (green).

## Reference
The reference to the paper will be available soon.
