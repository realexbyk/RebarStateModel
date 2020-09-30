import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import argparse
import imutils
from imutils.video import VideoStream
import pandas as pd

#pass all needed arguments to Yolo algorithm
options = {
    'model': 'cfg/tiny-yolo3-rebar-model.cfg',
    'load': 6125,
    'threshold': 0.2,
    'gpu': 0.75
}
tfnet = TFNet(options)



unt_color = (26, 0, 250)
td_color = (255, 0, 20)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time()
    ret, frame = capture.read()
    #if W is None or H is None:
        #(H, W) = frame.shape[:2]

    if ret:
        results = tfnet.return_predict(frame)
        for result in results:
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            confidence = result['confidence']
            label = result['label']
            if label == 'rebar_tied':
                color = td_color
            if label == "rebar_untied":
                color = unt_color
            text = '{}'.format(label)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
