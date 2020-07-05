# Smart Queuing System
Project 2 of the **Intel® Edge AI for IoT Developers** Nanodegree Program.

<img src="retail-queue.gif" alt="Smart Queueing System demo">

## The project
The aim of the project is to develop a _smart queuing system_ and choose the appropriate _hardware_ for three different scenarios:

- [Retail](scenarios/retail/README.md)
- [Manufacturing](scenarios/manufacturing/README.md)
- [Transportation](scenarios/transportation/README.md)

To meet the customer requirements and constraints for each scenario, the system is tested on CPU, Integrated GPU, VPU and FPGA. 

## Project steps
The project has been developed following the steps below.
### Choose best hardware
A best hardware choice has initially been determined based on the requirements and needs for each [scenarios](scenarios/README.md). 

The initial choice is documented in the [Proposal For Right Hardware Choice](choose-the-right-hardware-proposal-ashish-karn.pdf) document. 

### Build application
For the three scenario, the main script is `person_detect.py`. The detection model used is the pre-trained [person-detection-retail-0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html), based on MobileNetV2-like backbone. 

To test the script in the three different scenarios, use the following command:
```
python3 person_detect.py --model <path_to_the_model> --video ./scenarios/<scenario>/<scenario>_original.mp4 --queue_param ./scenarios/<scenario>/<scenario>_queue_param.npy
```

### Compare performance
The performance of the application has been tested using the _Udacity workspace_ provided with [IEI Tank AIOT Developer Kit](https://software.intel.com/content/www/us/en/develop/topics/iot/hardware/iei-tank-dev-kit-core.html). The `queue_job.sh` script is used to submit job to Intel DevCloud and then the result are collected once the job is finished.

The tested devices for each scenario are:
- **CPU**: [Intel® Core™ i5-6500TE Processor](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-)
- **Integrated GPU**: [Intel® HD Graphics 530](https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-)
- **VPU**: [Intel® Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick)
- **FPGA**: [IEI Mustang-F100-A10](https://www.ieiworld.com/mustang-f100/en/)


### Revise hardware choice
After testing the performance for each device, the initial hardware choice has been reviewed and both the testing and the revision are documented in the [Proposal For Right Hardware Choice](choose-the-right-hardware-proposal-ashish-karn.pdf) document.
