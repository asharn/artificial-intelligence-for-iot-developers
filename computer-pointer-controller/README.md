# Computer Pointer Controller

## Content 

- [Overview](#overview)
- [Demo video](#demo-video)
- [Project Set up and Installation](#project-setup-and-installation)
 - [Setup](#setup)
 - [Downloading Models Inference Files](#downloading-models-inference-files)
- [Arguments Documentation](#arguments-documentation)
- [Running the app](#running-the-app)
- [Directory Structure of the project](#directory-structure-of-the-project)
- [Benchmarks](#benchmarks)
- [Results](#results)
- [Edge Cases](#edge-cases)

## Overview
Computer Pointer Controller is an application that uses a gaze detection model to control the mouse pointer using an input video or a live stream from your webcam

## Demo video
[![Demo video](./demo/image_of_gaze.png)](./bin/demo.mp4)


## Project Setup and Installation

### Setup 

#### Install Intel® Distribution of OpenVINO™ toolkit
See this [guide](https://docs.openvinotoolkit.org/latest/) for installing openvino.

#### Intsalling pre-trained models

##### First, you have to initialize openVINO Environment 

* For linux:
```

user$ source /opt/intel/openvino/bin/setupvars.sh
[setupvars.sh] OpenVINO environment initialized

```

### Downloading Models Inference Files

- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html)
- [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

#### How to download the models

* Downloading Models 
for Face Detection Model

```
python <openvino directory>/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-0001"
```

for landmarks-regression-retail-0009

```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```

for head-pose-estimation-adas-0001

```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```

for gaze-estimation-adas-0002

```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```
## Arguments Documentation 

* project_file.py has several arguments
  * -h                : Get information about all the command line arguments
  * -fd               : (required) Specify the path of Face Detection model's name as shown below for specific precision "FP16"
  ```
  -fd "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
  ```
  * -fl               : (required) Specify the path of Facial landmarks Detection model's name as shown below for specific precision "FP16"
  ```
  -fl "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml"
  ```
  * -hp               : (required) Specify the path of hose pose Detection model's name as shown below for specific precision "FP16"
  ```
  -hp "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml"
  ```
  * -ge               : (required) Specify the path of gaze estimation model's name as shown below for specific precision "FP16"
  ```
  -ge "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml"
  ```
  * -i                : (required) Specify the path of input video file or enter cam for taking input video from webcam as shown below 
  ```
  -i "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/bin/demo.mp4"
  ```
  * -d                : (optional) Specify the target device to infer the video file on the model. Suppoerted devices are: CPU, GPU, FPGA (For running on FPGA used HETERO:FPGA,CPU), MYRIAD. 
  * -l                : (optional) Specify the absolute path of cpu extension if some layers of models are not supported on the device.
  * -pt               : (optional) Specify the probability threshold for face detection model to detect the face accurately from video frame.
  * -flag             : (optional) Specify the flags from fd, fl, hp, ge to visualize the output of corresponding models of each frame (write flags with space seperation. as shown below
  ```
  -flag fl fd ge
  ```
  
## Running the app

- Run on CPU 

```
python <project_file.py directory> -fd <Face detection model name directory> -fl <Facial landmark detection model name directory> -hp <head pose estimation model name directory> -ge <Gaze estimation model name directory> -i <input video directory> -d CPU
```


  For example:
  ```python3  "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/src/main.py" -fd "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml" -fl "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml" -hp "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml" -ge "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml" -i "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/bin/demo.mp4" -d CPU```
  

- Run on GPU 

```
python <project_file.py directory> -fd <Face detection model name directory> -fl <Facial landmark detection model name directory> -hp <head pose estimation model name directory> -ge <Gaze estimation model name directory> -i <input video directory> -d GPU
```

  For example:
  ```python3  "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/src/main.py" -fd "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml" -fl "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml" -hp "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml" -ge "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml" -i "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/bin/demo.mp4" -d GPU```
  

- Run on FPGA 

```
python <project_file.py directory> -fd <Face detection model name directory> -fl <Facial landmark detection model name directory> -hp <head pose estimation model name directory> -ge <Gaze estimation model name directory> -i <input video directory> -d HETERO:FPGA,CPU
```

  For example:
  ```python3  "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/src/main.py" -fd "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml" -fl "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml" -hp "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml" -ge "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml" -i "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/bin/demo.mp4" -d FPGA```
  

- Run on NSC2

```
python <project_file.py directory> -fd <Face detection model name directory> -fl <Facial landmark detection model name directory> -hp <head pose estimation model name directory> -ge <Gaze estimation model name directory> -i <input video directory> -d MYRIAD
```

  For example:
  ```python3  "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/src/main.py" -fd "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml" -fl "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml" -hp "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml" -ge "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml" -i "/home/ash/artificial-intelligence-for-iot-developers/computer-pointer-controller/bin/demo.mp4" -d MYRIAD```
  


## Directory Structure of the project 
<pre>
computer-pointer-controller/
├── bin
│   └── demo.mp4
├── intel
│   ├── face-detection-adas-0001
│   │   ├── FP16
│   │   │   ├── face-detection-adas-0001.bin
│   │   │   └── face-detection-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── face-detection-adas-0001.bin
│   │   │   └── face-detection-adas-0001.xml
│   │   └── FP32
│   │       ├── face-detection-adas-0001.bin
│   │       └── face-detection-adas-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP16-INT8
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP16-INT8
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── LICENSE
├── README.md
├── requirements.txt
├── resources
│   ├── FP16.png
│   ├── FP32.png
│   └── pipeline.png
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── main.py
    ├── mouse_controller.py
    ├── __pycache__
    │   ├── face_detection.cpython-37.pyc
    │   ├── facial_landmarks_detection.cpython-37.pyc
    │   ├── gaze_estimation.cpython-37.pyc
    │   ├── head_pose_estimation.cpython-37.pyc
    │   ├── input_feeder.cpython-37.pyc
    │   └── mouse_controller.cpython-37.pyc
    └── stats.txt
</pre>
- src folder contains all the source files:-
  * face_detection.py 
     - Contains preprocession of video frame, perform infernce on it and detect the face, postprocess the outputs.
     
  * facial_landmarks_detection.py
     - Take the deteted face as input, preprocessed it, perform inference on it and detect the eye landmarks, postprocess the outputs.
     
  * head_pose_estimation.py
     - Take the detected face as input, preprocessed it, perform inference on it and detect the head postion by predicting yaw - roll - pitch angles, postprocess the outputs.
     
  * gaze_estimation.py
     - Take the left eye, rigt eye, head pose angles as inputs, preprocessed it, perform inference and predict the gaze vector, postprocess the outputs.
     
  * input_feeder.py
     - Contains InputFeeder class which initialize VideoCapture as per the user argument and return the frames one by one.
     
  * mouse_controller.py
     - Contains MouseController class which take x, y coordinates value, speed, precisions and according these values it moves the mouse pointer by using pyautogui library.
  * main.py
     - Users need to run main.py file for running the app.
 
- bin folder contains demo video which user can use for testing the app and director structure image.

## Benchmarks
* I have Submited three jobs using this script to the DevCloud, using same demo video, but different hardware: 
  * IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE (CPU)
  * IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE (CPU + Integrated Intel® HD Graphics 530 card GPU)
  * IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE, with IEI Mustang-F100-A10 card (Arria 10 FPGA).

* for FP32
  | Type of Hardware | Total inference time in seconds              | Time for loading the model | fps |
  |------------------|----------------------------------------------|----------------------------|------
  | CPU              |  68                                          |  1.5                       |  9  |
  | GPU              |  69                                          |  55                        |  9  |
  | FPGA             |  118                                         |  6                         |  5  |

* for FP16
  | Type of Hardware | Total inference time in seconds              | Time for loading the model | fps |
  |------------------|----------------------------------------------|----------------------------|------
  | CPU              |  77                                          |  1.3                       |  8  |
  | GPU              |  75                                          |  52.4                      |  9  |
  | FPGA             |  125                                         |  4.5                       |  5  |


* for INT8
  | Type of Hardware | Total inference time in seconds              | Time for loading the model | fps |
  |------------------|----------------------------------------------|----------------------------|------
  | CPU              |  79                                          |  1.3                       |  8  |
  | GPU              |  74                                          |  52 .4                     |  9  |
  | FPGA             |  130                                         |  3                         |  5  |

## Results

- First of all, after decreasing prescison, accuracy of the model decreases
- As we see that GPA excutes more frames than the different hardwares, that goes the excution units and isntruction sets which is compatible and optmized with FP16
- FPGA takes higher inference time because it works on each gate and programmed it to be compatible for this application 


## Edge Cases 

- If there is more than one face detected, it extracts only one face and do inference on it and ignoring other faces.



