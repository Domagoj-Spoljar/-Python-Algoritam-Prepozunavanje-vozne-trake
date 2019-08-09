import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Image_processing_functions as IPF
import sys
import function_parameters as FP
import Lane_find_functions as LFF
import time
# from Lane_find_functions import *

def main():


    # ts = time.gmtime()
    # readable_time = time.strftime("%Y-%m-%d", ts)
    # print('time is: '+str(readable_time))

    # prefix='/home/profesor/Documents/CUlane/data/CULane'
    # save_folder='/home/profesor/Documents/advanced_lane_lanes/validation_images/'
    print('1.korak')
    f = open("train_gt.txt", "r+")
    ff = open("validation_groundtruth.txt", "w+")
    print('2.korak')
    allowed_folder='driver_182_30frame'

    for line in f:
    # line=f.readline()

        print('3.korak')

        # line=f.readline()
        print(line)
        parsed_line=line.split()
        # image_location=prefix+parsed_line[0]
        # print(image_location)

        file_name_array=parsed_line[0].split('/')
        if file_name_array[1] == allowed_folder:
            print('True')
            ff.write(line)
        # print(str(file_name_array[0]))
        print(str(file_name_array[1]))
        # print(str(file_name_array[2]))
        # print(str(file_name_array[3]))










    f.close()
    ff.close()
    return

###################################################################################################
if __name__ == "__main__":
    main()
