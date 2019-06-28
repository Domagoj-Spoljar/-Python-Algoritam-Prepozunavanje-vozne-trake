import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import function_parameters as FP
import time

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
filename = 'frame_count'
filename2 = 'calculated_binary_combinations'

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
        start = time.time()

        outfile = open(filename,'wb')
        pickle.dump(count,outfile)
        outfile.close()

        image = cv2.imread(image_folder+"frame%d.jpg" % count)     # save frame as JPEG file
        #if image is None or count > 300:                             # if image was not read successfully
        if image is None:                             # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            success = 0                                 # pause so user can see error message
      #success,image = vidcap.read()
        #imgOriginal=oszv.pipeline(image)
        infile = open(filename2,'rb')
        new_count = pickle.load(infile)
        infile.close()
        print('xdxdxdxdxdxd '+str(new_count))
        FP.binary_combinations=new_count
        processed_image =Lff.process_image_4lanes(image, FP.fullscreen)
        cv2.putText(processed_image, 'frame ' + str(count), (40,80), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        #processed_image = cv2.resize(processed_image,width,height)
        video.write(processed_image)
        count += 1
        end = time.time()
        # print('frames_to_video_dynamic time= '+str(end - start)+'sec')
        print('______________________________________')
        print('|    wrote a new frame: ', count,'   |',str(end - start)+'sec')
        print('______________________________________')
        f = open("fps_test_log.txt", "a")
        write_line=str(FP.video_tip)+' '+'frame:'+str(count)+' '+str(end - start)+' sec'+'\n'
        f.write(write_line)
    cv2.destroyAllWindows()
    video.release()
    f.close()

    return

###################################################################################################
if __name__ == "__main__":
    main()
