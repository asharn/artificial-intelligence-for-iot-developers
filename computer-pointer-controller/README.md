# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller

/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/src/main.py
/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml

python3  "/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/src/main.py" -fd "/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml" -fl "/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml" -hp "/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml" -ge "/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml" -i "/home/ash/Git repository/artificial-intelligence-for-iot-developers/computer-pointer-controller/bin/demo.mp4" -d CPU