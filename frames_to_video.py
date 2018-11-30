import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import Obrada_slike_za_video as oszv

video_name = 'bezveze_video2.mp4'
image_folder = './Test_images/dashcam_driving/'

frame = cv2.imread(image_folder+"frame1.jpg")
height, width, layers = frame.shape
print(frame.shape)
#video = cv2.VideoWriter(video_name, -1, 1, (width,height))
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))
success=1
count = 0

while success:
    image = cv2.imread(image_folder+"frame%d.jpg" % count)     # save frame as JPEG file
    if image is None:                             # if image was not read successfully
        print ("error: image not read from file \n\n")        # print error message to std out
        success = 0                                 # pause so user can see error message
  #success,image = vidcap.read()
    imgOriginal=oszv.pipeline(image)
    video.write(imgOriginal)
    print('wrote a new frame: ', success)
    count += 1

cv2.destroyAllWindows()
video.release()
