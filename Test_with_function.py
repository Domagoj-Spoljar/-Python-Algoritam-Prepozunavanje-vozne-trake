import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import Image_processing_functions as IPF
import sys

def main():

    #dashcam_image_path = './Test_images/dashcam_driving/'
    dashcam_image_path = './Test_images/challnege_video/'
    #dashcam_image_path = './Test_images/harder_challenge_video/'
    #dashcam_image_path = './Test_images/project_video/'
    img_arg="frame"
    #count = 138
    #count = 139
    #count = 170
    #count = 60
    #count = 290
    count = 0
    #count = 822

    k=0
    #cv2.namedWindow('prikaz', cv2.WINDOW_NORMAL)
    while k is not 27:

        imgOriginal = cv2.imread(dashcam_image_path+img_arg+str(count)+".jpg")               # open image
        #cv2.imshow(img_arg+str(count)+".jpg", imgOriginal)
        cv2.namedWindow(img_arg+str(count)+".jpg", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(img_arg+str(count)+".jpg",1280,720)
        if imgOriginal is None:                             # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return

        processed_image =Lff.process_image_4lanes(imgOriginal,fullscreen=False)

#--------------------------------------------------------------------------------------------------------------------
        cv2.imshow(img_arg+str(count)+".jpg", processed_image)
        #print(final_image.shape)
        #cv2.imwrite('Output_'+img_arg+str(count)+".jpg",processed_image)
        k = cv2.waitKey()
        if k == 83:
            count=count+1
        elif k == 81:
            if count !=0:
                count=count-1# hold windows open until user presses a key
        cv2.destroyAllWindows()                     # remove windows from memory
    cv2.destroyAllWindows()
    return

###################################################################################################
if __name__ == "__main__":
    main()
