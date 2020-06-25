# Artificial Intelligence For IoT Developers
This repo will contain different AI applied within IOT application especially created for intel distribution.
Projects are listed below:
To design, test, and deploy an edge AI application in any number of settings for a broad of array of different companies, industries, and devices.

Content to view all the projects are listed below:
  1. [**Deploy a People Counter App at the Edge**](https://github.com/asharn/artificial-intelligence-for-iot-developers/tree/master/people-counter-app#deploy-a-people-counter-app-at-the-edge)
  2. [**Smart Queuing System**](https://github.com/asharn/artificial-intelligence-for-iot-developers/tree/master/smart-queuing-system)
  3. [**Deploy a People Counter App at the Edge**](https://github.com/asharn/artificial-intelligence-for-iot-developers/tree/master/people-counter-app#deploy-a-people-counter-app-at-the-edge)


  - [**Deploy a People Counter App at the Edge**](https://github.com/asharn/artificial-intelligence-for-iot-developers/tree/master/people-counter-app#deploy-a-people-counter-app-at-the-edge)\
![people-counter-python](./people-counter-app/images/people-counter-image.png)
   ### What it Does
   The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.

   ### How it Works
   The counter will use the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used should be able to identify people in a video frame. The app should count the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.
   You will choose a model to use and convert it with the Model Optimizer.
![architectural diagram](./people-counter-app/images/arch_diagram.png)
