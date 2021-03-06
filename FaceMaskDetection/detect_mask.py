# pip install pygame
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from os.path import dirname, join
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from pygame import mixer


def detect_mask(frame, faceDet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceDet.setInput(blob)
    detections = faceDet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []

    # face and mask detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # bounding boxes
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # at least a single face to predict
    if len(faces) > 0:
        faces = np.array(faces, dtype="uint8")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)


# load our serialized face detector model from disk
#https://github.com/pourabkarchaudhuri/face-detection/tree/master/caffe
prototxtPath = r"deploy.protext"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceDet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

#  start video stream
print("Opening the camera...")
vs = VideoStream(src=0).start()

# loop over the images from the video stream
j = 0
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_mask(frame, faceDet, maskNet)

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # draw bounding box and probability text with different colors
        label = "Mask" if mask > withoutMask else "No Mask"
        if label == "No Mask":
            color = (0, 0, 255)
            j = j + 1
            # evaluate few second for confirmation of no mask and play announcement
            if j == 15:
                mixer.init()
                mixer.music.load('Recording.mp3')
                mixer.music.play()
                print("No mask detected")
                time.sleep(10.5)
                j = 0
        else:
            color = (0, 255, 0)
            j = 0

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # the label and rectangle on output stream
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 5)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'e' key to exit
    if key == ord("e"):
        break

cv2.destroyAllWindows()
vs.stop()

