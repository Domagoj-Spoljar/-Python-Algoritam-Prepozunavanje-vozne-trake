import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import function_parameters as FP


# video_name = 'test_video_4lanes_1.13.mp4'
# image_folder = './Test_images/dashcam_driving/'

# video_name = 'challenge_video_4lanes_1.8.mp4'
# image_folder = './Test_images/challnege_video/'

# video_name = 'harder_challenge_video_4lanes_1.8_fullscreen.mp4'
# image_folder = './Test_images/harder_challenge_video/'
#
# video_name = 'project_video_4lanes_1.11_confidence.mp4'
# image_folder = './Test_images/project_video/'

count=0
def main():
    global count
    video_name = FP.video_name
    image_folder = FP.dashcam_image_path

    frame = cv2.imread(image_folder+"frame1.jpg")
    height, width, layers = frame.shape

    # fullscreen=False
    if FP.fullscreen is False:
        height,width=960,1280
    else:
        height,width=720,1280


    print(frame.shape)
    #video = cv2.VideoWriter(video_name, -1, 1, (width,height))
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))
    success=1
    # count = 215
    count = 0

    while success:
        FP.calibration_frame=count
        image = cv2.imread(image_folder+"frame%d.jpg" % count)     # save frame as JPEG file
        #if image is None or count > 300:                             # if image was not read successfully
        if image is None:                             # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            success = 0                                 # pause so user can see error message
      #success,image = vidcap.read()
        #imgOriginal=oszv.pipeline(image)
        processed_image =Lff.process_image_4lanes(image, FP.fullscreen)
        cv2.putText(processed_image, 'frame ' + str(count), (40,80), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        #processed_image = cv2.resize(processed_image,width,height)
        video.write(processed_image)
        print('wrote a new frame: ', count)
        count += 1

    cv2.destroyAllWindows()
    video.release()

    return

###################################################################################################
if __name__ == "__main__":
    main()
