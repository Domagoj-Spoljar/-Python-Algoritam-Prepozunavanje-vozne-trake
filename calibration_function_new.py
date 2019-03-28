import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import Image_processing_functions as IPF
import sys
import function_parameters as FP

def main(count):

    if len(sys.argv) == 2:
        count=sys.argv[1]
        print('Processing image: frame'+count)
    else:
        # count=str(FP.frame)
        # count=str(FP.calibration_frame)
        print('Processing default image: frame'+count)

    dashcam_image_path = FP.dashcam_image_path

    imgOriginal = cv2.imread(dashcam_image_path+'frame'+str(count)+".jpg")              # open image
    #cv2.imshow(img_arg+str(count)+".jpg", imgOriginal)
    #cv2.namedWindow('frame'+str(count)+".jpg", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('frame'+str(count)+".jpg",1280,720)
    if imgOriginal is None:                             # if image was not read successfully
        print ("error: image not read from file \n\n")        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return

    #processed_image =Lff.process_image_4lanes(imgOriginal,fullscreen=False)
    img_undistort = IPF.undistort(imgOriginal)

    #get points on image for perspective transform
    h,w = imgOriginal.shape[:2]
    src,dst = FP.unwarp_points(h,w)

    # Perspective Transform
    img_unwarp, M, Minv = IPF.unwarp(img_undistort, src, dst)
    # first,second=IPF.calibrate_IPF(img_unwarp)
    first,second=IPF.calibrate_IPF_yellow_white(img_unwarp,sobel=True)
    # print(str(first[0][1]))
    # print(str(second[0][1]))
    # lista=[first[0][1],second[0][1]]
    # FP.binary_combinations=lista
    # print(len(FP.calibrated_combinations))
    FP.calibrated_combinations=[first,second]
    print(FP.calibrated_combinations)
    # FP.binary_combinations[1]=second[0][1]
    # print(FP.calibrated_combinations[0])
    # print(FP.calibrated_combinations[0][1])
    # print(FP.calibrated_combinations[0][0][1])
    # print(FP.calibrated_combinations[0][1][1])
    # print(FP.calibrated_combinations[0][2][1])
    # print(len(FP.calibrated_combinations))
#--------------------------------------------------------------------------------------------------------------------
    return first,second

###################################################################################################
if __name__ == "__main__":
    main()
