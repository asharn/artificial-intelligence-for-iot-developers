"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import faulthandler;

faulthandler.enable()
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# from sklearn.cluster import KMeans

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


"""handling input stream"""


def handling_input_stream(input_mode, single_image_mode):
    """If input is an image"""
    if input_mode.endswith('.jpg') or input_mode.endswith('.bmp'):
        single_image_mode = True
        input_source = input_mode
    elif input_mode == 'CAM':
        input_source = 0
    else:
        input_source = input_mode
    return input_source, single_image_mode


"""Get the average color of the input image"""


def Get_Average_Color(imageCropped):
    averageColor = imageCropped.mean(axis=0).mean(axis=0)
    return averageColor


"""draw bounding boxes"""


def draw_boxes(frame, result, args, width, height, prevAverageColor, personEnteredFlag, personStateOut, startTime,
               duration, stopPerson):
    '''
    Draw bounding boxes onto the frame.
    '''
    current_count = 0
    currentAverageColor = prevAverageColor
    """Looping over the results"""
    for box in result[0][0]:  # Output shape is 1x1x100x7
        #get threshold of given boundary box
        conf = box[2]
        #get label class of the result when it equals to one ( human label ), the condition becomes true
        if box[1] == 1:
            #if the threshold is bigger than or equal to desired threshold
            if conf >= args.prob_threshold:
                #Pre-process the output
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                #Draw boundary box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
                if (xmin < width) and (xmax < width) and (ymin < height) and (ymax < height):
                    #get the average color of the boundary box and do some operations on it
                    imgCropped = frame[xmin:xmax, ymin:ymax]
                    currentAverageColor = Get_Average_Color(imgCropped)
                    difference = np.absolute(np.subtract(currentAverageColor, prevAverageColor))
                    difference = difference.mean(axis=0)
                    if np.isnan((difference)) == False:
                        difference = int(difference * 10)
                        if np.any(difference > 20):
                            if personEnteredFlag == False:
                                personEnteredFlag = True
                                current_count = current_count + 1
                            else:
                                pass

                        if np.all(difference == 0) and personEnteredFlag == True:

                            if stopPerson == False:
                                personEnteredFlag = False
                                personStateOut = False
                                stopPerson = True

                        if np.any(difference > 1) and np.any(difference < 11):
                            if personStateOut == False:
                                personStateOut = True
                                stopPerson = False

                            else:
                                pass

    return frame, current_count, currentAverageColor, personEnteredFlag, personStateOut, startTime, duration, stopPerson


def preprocess_input(frame, input_height, input_width):
    pre_frame = cv2.resize(frame, (input_height, input_width))
    pre_frame = pre_frame.transpose((2, 0, 1))
    pre_frame = pre_frame.reshape(1, *pre_frame.shape)
    return pre_frame


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, port=MQTT_PORT, keepalive=MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    current_average_color = 0
    person_entered_flag = 0
    person_state_out = False
    start_time = 0
    duration = 0
    lastPeopleCount = 0
    total_people_count = 0
    stopPerson = True
    prevDuration = 0
    flag_zero_count = False
    current_value = 0
    prev_value = 0
    counter_inter = 0
    single_image_mode = False
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    input_source, single_image_mode = handling_input_stream(args.input, single_image_mode)
    cap = cv2.VideoCapture(input_source)
    cap.open(input_source)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    counter = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()

        key_pressed = cv2.waitKey(60)
        if not flag:
            sys.stdout.flush()
            break
        displayFrame = frame.copy()
        ### TODO: Pre-process the image as needed ###
        pre_frame = preprocess_input(frame, net_input_shape[3], net_input_shape[2])

        if counter == 0:
            RoomWithNoBodyAverage = Get_Average_Color(frame)
            RoomWithNoBodyAverage = RoomWithNoBodyAverage.mean(axis=0)
            counter = 1
        ### TODO: Start asynchronous inference for specified request ###
        interferenceTimeStart = time.time()
        infer_network.exec_net(pre_frame)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            detectedTimeStart = time.time() - interferenceTimeStart
            counter_inter = counter_inter + 1
            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            displayFrame, current_count, current_average_color, person_entered_flag, person_state_out, start_time, duration, stopPerson = draw_boxes(
                displayFrame, result, args, width, height, current_average_color, person_entered_flag, person_state_out,
                start_time, duration, stopPerson)
            inf_time_message = "Inference time: {:.3f}ms".format(detectedTimeStart * 1000)
            cv2.putText(displayFrame, inf_time_message, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 30, 30), 1)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            AverageNow = Get_Average_Color(frame)
            AverageNow = AverageNow.mean(axis=0)
            current_value = int((np.abs(RoomWithNoBodyAverage / AverageNow)) * 10000)

            # when someone exits the frame
            if current_count == 0 and current_value > 10017 and prev_value < 10015 and person_entered_flag == False:
                if flag_zero_count == False:
                    value555 = (counter_inter * detectedTimeStart) / 2
                    duration = int(time.time() - start_time - value555)
                    client.publish("person/duration", json.dumps({"duration": duration}))
                    flag_zero_count = True
                    client.publish("person", json.dumps({"count": int(current_count)}))

            prev_value = current_value

            # client.publish("person", json.dumps({"count": int(lastPeopleCount)}))
            # publishing the current count
            if flag_zero_count == True and person_entered_flag == False:
                client.publish("person", json.dumps({"count": int(current_count)}))
            else:
                flag_zero_count = False
                client.publish("person", json.dumps({"count": int(lastPeopleCount)}))

            # When Someone enters the frame
            if person_entered_flag == True and current_count != 0:
                if duration != 0 and duration != prevDuration and total_people_count >= 1:
                    prevDuration = duration

                counter_inter = 0
                start_time = time.time()
                if lastPeopleCount == current_count:
                    if lastPeopleCount != 0:
                        lastPeopleCount = lastPeopleCount - 1
                total_people_count = total_people_count + current_count - lastPeopleCount
                lastPeopleCount = current_count
                client.publish("person", json.dumps({"total": int(total_people_count)}))
            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(displayFrame)
        ### TODO: Write an output image if `single_image_mode` ##
        if single_image_mode:
            cv2.imwrite('outputImage.jpg', frame)

    client.publish("person", json.dumps({"count": int(0)}))
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()