# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

TensorFlow Object Detection Model Zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) contains many pre-trained models on the coco dataset. 
ssd_inception_v2_coco and faster_rcnn_inception_v2_coco performed good as compared to rest of the models, 
but, in this project, ssd_inception_v2_coco is used which is fast in detecting people with less errors. 
Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo..

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were drastically change. If I take example of the ssd_inception_v2_coco then its speed is 31 ms before IR conversion but after conversion it was less tha 31 ms.




## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following two models:

- Model 1: Faster_rcnn_inception_v2_coco_2018_01_28
  - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments and commands '''python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json'''
  - The model was insufficient for the app because each time I was trying to run app it was showing segmentation fault.
  - I tried to improve the model for the app by using different theashold but result where not fruitful.

- Model 2: ssd_mobilenet_v2_coco_2018_03_29
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments of the command '''python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json'''
  - The model was sufficiant for the app because it was able to recognize accuratly people.
  - I tried to improve the model for the app by using different theashold and found that there was accuracy increase with duration of time also.
  
  Below is the screenshot which will show the runnning condition of the application.
  
  ![First Person](screenshots/image1.png)\
  ![Second Person](screenshots/image2.png)\
  ![Third Person](screenshots/image3.png)\
  ![Third Person](screenshots/image4.png)\
  ![Fourth Person](screenshots/image5.png)



  
  
