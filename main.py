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
import os.path as path
import socket
import json
import cv2
import re
import numpy as np

import logging as log
import paho.mqtt.client as mqtt
import urllib.request as urllib

from argparse import ArgumentParser
from inference import Network
from boundingbox_detection import BoundingBox, BoundingBoxTracker

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
# IMAGE_WIDTH = 640
# IMAGE_HEIGHT = 360
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MODEL_XML = "model/frozen_inference_graph.xml"
LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"
LABELS_FILE = "model/labels.pbtxt"

# Create the dictionaries used form ID <-> Label conversions
IDFromLabel = {}
labelFromID = {}


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
    parser.add_argument("-cam", "--camera", type=bool, default=False,
                        help="Capture from live camera instead of video file")
    parser.add_argument("-s", "--preview", type=bool, default=False,
                        help="Show preview image window")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Output file with the processed content")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def get_boxes(result, prob_threshold=0.30):
    expected_label = 'person'
    expected_type = IDFromLabel[expected_label]
    bbs = []
    for box in result[0][0]:  # Output shape is 1x1x100x7
        label_id = int(box[1])
        conf = box[2]
        if label_id == expected_type and conf >= prob_threshold:
            new_box = BoundingBox(box[3], box[4], box[5], box[6], expected_label, conf)
            bbs.append(new_box)
        else:
            if label_id > 0: '''
                print('Detecting: ' + labelFromID[label_id] + ' with probability: ' + str(conf))
                '''
    return bbs


def draw_boxes(frame, bounding_boxes, width, height):
    # print(bounding_boxes)
    for key in bounding_boxes:
        box = bounding_boxes[key]
        xmin = int(box.xmin * width)
        ymin = int(box.ymin * height)
        xmax = int(box.xmax * width)
        ymax = int(box.ymax * height)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "PERSON: " + str(box.id), (xmin + 25, ymin + 25), font, 0.50, (0, 0, 255), 2)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    return frame


def open_stream(args):
    # Check if the input is a webcam
    from_camera = args.camera

    if from_camera:
        cap = cv2.VideoCapture(0)
        #print("Camera capture resolution: (" + str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) + " x " +
        #      str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) + ")")
    else:
        if not hasattr(args, 'input'):
        #    print('Input file must be specified')
            exit(-1)
        cap = cv2.VideoCapture(args.input)
        cap.open(args.input)

    # Create a video writer for the output video
    if args.out:
        try:
            os.remove('out.mp4')
        except FileNotFoundError: '''
            print('previous ' + args.out + ' not found')
            '''

    source_width = IMAGE_WIDTH
    source_height = IMAGE_HEIGHT

    if cap.isOpened():
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #print("source image size: ( " + str(source_width) + " x " + str(source_height) + " )")

    out = None
    if args.out:
        out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'), 15, (source_width, source_height))

    if not cap.isOpened():
        #print('Could not open input')
        cap.release()
        if out:
            out.release()

    return cap, out, source_width, source_height
    # Process frames until the video ends, or process is exited


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(MODEL_XML)

    input_shape = infer_network.get_input_shape()
    output_shape = infer_network.get_output_shape()
    #print("Input shape: ", input_shape)
    #print("Output shape: ", output_shape)
    width = input_shape[2]
    height = input_shape[3]
    #print('Input image will be resized to (' + str(width) + ' x ' + str(height) + ') for inference')

    bb_tracker = BoundingBoxTracker(prob_threshold, 3, 20)

    ### TODO: Handle the input stream ###
    cap, out, source_width, source_height = open_stream(args)

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        frame_inference = cv2.resize(frame, (width, height))

        # Transform the image from the (300, 300, 3) original size to the (1, 3, 300, 300) input shape
        frame_inference = frame_inference.transpose((2, 0, 1))
        frame_inference = frame_inference.reshape(1, *frame_inference.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(frame_inference)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            bboxes = get_boxes(result, prob_threshold)
            bb_tracker.updateBBs(bboxes)
            frame_processed = draw_boxes(frame, bb_tracker.getBBs(), source_width, source_height)

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        cv2.namedWindow('preview')
        cv2.imshow('preview', frame_processed)
        if out:
            out.write(frame_processed)
        else:
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()

        ### TODO: Send the frame to the FFMPEG server ###

        # Break if escape key pressed
        if key_pressed == 27:
            break

        ### TODO: Write an output image if `single_image_mode` ###

    if out:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


def load_labels():
    # Grab labels from Mobilenet V2 COCO
    if not path.exists(LABELS_FILE):
        urllib.urlretrieve(LABELS_URL, LABELS_FILE)

    # Parse file and create dictionaries
    with open(LABELS_FILE) as f:
        txt = f.read()
    lines = txt.split('\n')
    i = 1
    while i < len(lines):
        id = int(re.search('.+: (.*)', lines[i + 1], re.IGNORECASE).group(1))
        display_name = re.search('.+: \"(.*)\"', lines[i + 2], re.IGNORECASE).group(1)
        i += 5
        # print(str(id) + ': ' + display_name)
        IDFromLabel[display_name] = id
        labelFromID[id] = display_name
    #print("Labels loaded")


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    load_labels()
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    client.disconnect()


if __name__ == '__main__':
    main()
