#!/usr/bin/env python
import pyrealsense2 as rs
import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util


def init_object_detection(class_to_remove):

    ########## end pyrealsense2 ##################
    return pipeline, interpreter, input_details, output_details, freq, frame_rate_calc, colors_hash, width, height, min_conf_threshold


def espeak(texttospeak):
    s = '"{input_string}"'.format(input_string=texttospeak)
    os.system('espeak -s 150 ' + s + ' --stdout > audio.wav')


def detect_objects(class_to_remove, pipeline, interpreter, input_details, output_details, freq, frame_rate_calc, colors_hash, width, height, min_conf_threshold, labels):
    # This call waits until a new coherent set of frames is available on a device
    # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
    for num_frames in range(int(5)):
        t1 = cv2.getTickCount()
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        scaled_size = (color_frame.width, color_frame.height)
        scaled_frame = cv2.resize(
            color_image, (width, height))
        # expand the image
        input_data = np.expand_dims(scaled_frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # Retrieve detection results
        # Bounding box coordinates of detected objects
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[
            0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[
            0]  # Confidence of detected objects
        num = interpreter.get_tensor(output_details[3]['index'])[0]

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        # for index, label_name in enumerate(labels):
        #    print(label_name, index)

        # if not depth:
        #    continue
        # dist = depth.get_distance(int(640/2), int(480/2))
        # print(dist)

        objects_info = []

        for i in range(int(num)):
            class_ = classes[i]
            score = scores[i]
            box = boxes[i]
            if class_ not in colors_hash:
                colors_hash[class_] = tuple(
                    np.random.choice(range(256), size=3))
            # min_conf_threshold:
            if score > min_conf_threshold and classes[i] not in class_to_remove:
                left = int(box[1] * color_frame.width)
                top = int(box[0] * color_frame.height)
                right = int(box[3] * color_frame.width)
                bottom = int(box[2] * color_frame.height)
                xmin = left
                p1 = (left, top)
                p2 = (right, bottom)
                # draw box
                r, g, b = colors_hash[class_]
                cv2.rectangle(color_image, p1, p2,
                              (int(r), int(g), int(b)), 2, 1)
                # Draw Score Label
                # Look up object name from "labels" array using class index
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(
                    scores[i]*100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                # Make sure not to draw label too close to top of window
                label_ymin = max(top, labelSize[1] + 10)
                # Draw white box to put label text in
                cv2.rectangle(color_image, (xmin, label_ymin-labelSize[1]-10), (
                    xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(color_image, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

                # Draw Distance Label
                box_center = (int((left + right)/2), int((top + bottom)/2))
                # box_center = (int(color_frame.width/2),
                #              int(color_frame.height/2))
                # Look up object name from "labels" array using class index
                object_distance = depth.get_distance(
                    int((left + right)/2), int((top + bottom)/2))

                # label_dist = 'Distance: %f' % object_distance
                label_dist = "Distance: {distance}m".format(
                    distance=round(object_distance, 3))
                labelSize, baseLine = cv2.getTextSize(
                    label_dist, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                # Make sure not to draw label too close to top of window
                label_ymin = max(bottom, labelSize[1] - 10)
                # Draw white box to put label text in
                cv2.rectangle(color_image, (xmin, label_ymin-labelSize[1]-10), (
                    xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(color_image, label_dist, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

                # Draw Box CenterPoint
                cv2.circle(color_image, box_center, 7, (255, 255, 255))

                if box_center[0] < color_frame.width*(1/3):
                    direction = "Left"
                elif box_center[0] > color_frame.width*(2/3):
                    direction = "Right"
                else:
                    direction = "Center"

                objects_info.append((object_name, object_distance, direction))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.putText(color_image, 'FPS: {0:.2f}'.format(
            frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('RealSense', color_image)
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        cv2.waitKey(1)
        objects_info.sort(key=lambda tup: tup[1])
        # os.system('clear')
        print([(object[0] + " is " + str(round(object[1], 1)) + " meters and " + object[2])
               for object in objects_info])
    return objects_info


def main():
    ##########   tflite     ##################
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    # resW, resH = args.resolution.split('x')
    imW, imH = int(640), int(480)

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument

    interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    ##########  end tflite  ##################
    ########## pyrealsense2 ##################
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()

    # Configure streams
    config = rs.config()
    config.enable_stream(rs.stream.color, imW, imH)
    config.enable_stream(rs.stream.depth, imW, imH)
    # Start streaming
    pipeline.start(config)
    colors_hash = {}
    class_to_remove = {1,  2,  3,  4,  5,  6,  7,  8,
                       9, 10, 11, 12, 13, 14, 15, 16, 17,
                       18, 19, 20, 21, 22, 23, 24, 25,
                       26, 27, 28, 29, 30, 31, 32, 33, 34,
                       35, 36, 37, 38, 40, 41, 42, 43, 44,
                       46, 47, 48, 49, 50, 51, 52, 53, 54,
                       55, 56, 57, 58, 59, 61, 74, 75, 76,
                       77, 78, 79}
    try:

        objects_info = detect_objects(class_to_remove, pipeline, interpreter, input_details,
                                      output_details, freq, frame_rate_calc, colors_hash, width, height, min_conf_threshold, labels)
        espeak([(object[0] + " is " + str(round(object[1], 1)) + " meters and " + object[2])
               for object in objects_info])
        cv2.destroyAllWindows()

        exit(0)

    except Exception as e:
        print(e)
    pass


if __name__ == "__main__":
    main()
