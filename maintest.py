# Easy Machine Learning & Object Detection with Teachable Machine
#
# Michael D'Argenio
# mjdargen@gmail.com
# https://dargenio.dev
# https://github.com/mjdargen
# Created: February 6, 2020
# Last Modified: February 13, 2021
#
# This program uses Tensorflow and OpenCV to detect objects in the video
# captured from your webcam. This program is meant to be used with machine
# learning models generated with Teachable Machine.
#
# Teachable Machine is a great machine learning model trainer and generator
# created by Google. You can use Teachable Machine to create models to detect
# objects in images, sounds in audio, or poses in images. For more info, go to:
# https://teachablemachine.withgoogle.com/
#
# For this project, you will be generating a image object detection model. Go
# to the website, click "Get Started" then go to "Image Project". Follow the
# steps to create a model. Export the model as a "Tensorflow->Keras" model.
#
# To run this code in your environment, you will need to:
#   * Install Python 3 & library dependencies
#       * Follow instructions for your setup
#   * Export your teachable machine tensorflow keras model and unzip it.
#       * You need both the .h5 file and labels.txt
#   * Update model_path to point to location of your keras model
#   * Update labels_path to point to location of your labels.txt
#   * Adjust width and height of your webcam for your system
#       * Adjust frameWidth with your video feed width in pixels
#       * Adjust frameHeight with your video feed height in pixels
#   * Set your confidence threshold
#       * conf_threshold by default is 90
#   * If video does not show up properly, use the matplotlib implementation
#       * Uncomment "import matplotlib...."
#       * Comment out "cv2.imshow" and "cv2.waitKey" lines
#       * Uncomment plt lines of code below
#   * Run "python3 tm_obj_det.py"

import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import pyttsx3
import math
import os
# use matplotlib if cv2.imshow() doesn't work
# import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from color import colorDetection

cred = credentials.Certificate(
    './pet007-1c21d-firebase-adminsdk-b6v3j-9777e17f22.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://pet007-1c21d-default-rtdb.asia-southeast1.firebasedatabase.app/'
})
print('firebase connected')
ref = db.reference('PetStatus')
print('firebase realtime connected')

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


# this process is purely for text-to-speech so it doesn't hang processor
def speak(speakQ, ):
    # initialize text-to-speech object
    engine = pyttsx3.init()
    # can adjust volume if you'd like
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume)  # add number here
    # initialize last_msg to be empty
    last_msg = ""
    # keeps program running forever until ctrl+c or window is closed
    while True:
        msg = speakQ.get()
        # clear out msg queue to get most recent msg
        while not speakQ.empty():
            msg = speakQ.get()
        # if most recent msg is different from previous msg
        # and if it's not "Background"
        if msg != last_msg and msg != "Background":
            last_msg = msg
            # text-to-speech say class name from labels.txt
            engine.say(msg)
            engine.runAndWait()
        if msg == "Background":
            last_msg = ""


def main():

    # read .txt file to get labels
    labels_path = f"{DIR_PATH}./labels.txt"
    # open input file label.txt
    labelsfile = open(labels_path, 'r')

    # initialize classes and read in lines until there are no more
    classes = []
    line = labelsfile.readline()
    while line:
        # retrieve just class name and append to classes
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    # close label file
    labelsfile.close()

    # load the teachable machine model
    model_path = f"{DIR_PATH}./keras_model.h5"
    model = tf.models.load_model(model_path, compile=False)

    # initialize webcam video object
    cap = cv2.VideoCapture(1)

    # width & height of webcam video in pixels -> adjust to your size
    # adjust values if you see black bars on the sides of capture window
    frameWidth = 720
    frameHeight = 480

    # set width and height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # enable auto gain
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # creating a queue to share data to speech process
    speakQ = multiprocessing.Queue()

    # creating speech process to not hang processor
    p1 = multiprocessing.Process(target=speak, args=(speakQ, ), daemon="True")

    # starting process 1 - speech
    p1.start()

    # keeps program running forever until ctrl+c or window is closed
    while True:

        # disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model.
        # We are inputting 1x 224x224 pixel RGB image.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # capture image
        check, frame = cap.read()

        # mirror image - mirrored by default in Teachable Machine
        # depending upon your computer/webcam, you may have to flip the video
        # frame = cv2.flip(frame, 1)

        # crop to square for use with TM model
        margin = int(((frameWidth-frameHeight)/2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # resize to 224x224 for use with TM model
        resized_img = cv2.resize(square_frame, (224, 224))
        # convert image color to go to model
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        # turn the image into a numpy array
        image_array = np.asarray(model_img)
        # normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # load the image into the array
        data[0] = normalized_image_array

        # run the prediction
        predictions = model.predict(data)

        # confidence threshold is 90%.
        conf_threshold = 90
        confidence = []
        conf_label = ""
        threshold_class = ""
        # create blach border at bottom for labels
        per_line = 2  # number of classes per line of text
        bordered_frame = cv2.copyMakeBorder(
            square_frame,
            top=0,
            bottom=30 + 15*math.ceil(len(classes)/per_line),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # for each one of the classes
        for i in range(0, len(classes)):
            # scale prediction confidence to % and apppend to 1-D list
            confidence.append(int(predictions[0][i]*100))
            # put text per line based on number of classes per line
            if (i != 0 and not i % per_line):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # append classes and confidences to text for label
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            # prints last line
            if (i == (len(classes)-1)):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # if above confidence threshold, send to queue
            if confidence[i] > conf_threshold:
                speakQ.put(classes[i])
                threshold_class = classes[i]
        # add label class above confidence threshold
        cv2.putText(
            img=bordered_frame,
            text=threshold_class,
            org=(int(0), int(frameHeight+20)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255)
        ) 
        if threshold_class == "Pet Sleep":
            ref.update({
                'Status': 'สัตว์เลี้ยงกำลังนอน'
            })
        if threshold_class == "Pet Walk":
            ref.update({
                'Status': 'สัตว์เลี้ยงกำลังเดิน'
            })

        # original video feed implementation
        cv2.imshow("Capturing", bordered_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # # if the above implementation doesn't work properly
        # # comment out two lines above and use the lines below
        # # will also need to import matplotlib at the top
        # plt_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB)
        # plt.imshow(plt_frame)
        # plt.draw()
        # plt.pause(.001)

    # terminate process 1
    p1.terminate()


def colorDetection():
    # Capturing video through webcam
    webcam = cv2.VideoCapture(0)

    # Start a while loop
    while(1):

        # Reading the video from the
        # webcam in image frames
        _, imageFrame = webcam.read()

        # Convert the imageFrame in
        # BGR(RGB color space) to
        # HSV(hue-saturation-value)
        # color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

        # Set range for red color and
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # Set range for green color and
        # define mask
        green_lower = np.array([25, 52, 72], np.uint8)
        green_upper = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # Set range for blue color and
        # define mask
        blue_lower = np.array([94, 80, 2], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame,
                                  mask=red_mask)

        # For green color
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                    mask=green_mask)

        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask=blue_mask)

        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(red_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 255), 2)

                cv2.putText(imageFrame, "Red Colour", (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (0, 0, 255))

        # Creating contour to track green color
        contours, hierarchy = cv2.findContours(green_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 255, 0), 2)

                cv2.putText(imageFrame, "Green Colour", (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 255, 0))

        # Creating contour to track blue color
        contours, hierarchy = cv2.findContours(blue_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (255, 0, 0), 2)

                cv2.putText(imageFrame, "Blue Colour", (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (255, 0, 0))

        # Program Termination
        cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
    colorDetection()
