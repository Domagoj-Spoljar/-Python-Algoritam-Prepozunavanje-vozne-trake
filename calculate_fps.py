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

def main():

    frame_count=0
    sec_list=[]
    sec_accumulated=0

    file_name='fps_test_log.txt'
    f = open(file_name, "r")

    for line in f:
    # line=f.readline()


        # line=f.readline()
        # print(line)
        parsed_line=line.split()
        frame_time=parsed_line[2]
        # print(frame_time)
        frame_count+=1
        sec_list.append(float(frame_time))
        sec_accumulated+=float(frame_time)



    f.close()
    average_fps_acc=sec_accumulated/frame_count
    average_fps=np.sum(sec_list)/frame_count
    print('average frame count [acc]: '+str(1/average_fps_acc)+' fps')
    print('average frame count [org]: '+str(1/average_fps)+' fps')


    return

###################################################################################################
if __name__ == "__main__":
    main()
