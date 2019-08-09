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

    TP_count=0
    FP_count=0
    TN_count=0
    FN_count=0
    threshold_value=0.2

    folder_fields=[['05312333_0003.MP4','05312336_0004.MP4','06010018_0018.MP4','06010552_0049.MP4','06010734_0083.MP4','06010904_0113.MP4','06010907_0114.MP4','06010910_0115.MP4','06010913_0116.MP4','06011523_0209.MP4','06011526_0210.MP4','06011529_0211.MP4','06011535_0213.MP4','06011611_0225.MP4','06011929_0291.MP4'],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #TP
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #FP
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #TN
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] #FN


    # f = open("validation-selected-result-2019-07-08.txt", "r")
    f = open("validation-selected-result-2019-07-09.txt", "r")

    # line=f.readline()
    # print(line)
    # parsed_line=line.split()
    #
    # print(parsed_line[2])
    # print(parsed_line[3])
    # print(parsed_line[4])
    # print(parsed_line[5])
    # print('')
    # print(parsed_line[7])
    # print(parsed_line[8])
    # print(parsed_line[9])
    # print(parsed_line[10])
    # print('')
    # print(parsed_line[12])
    # print(parsed_line[13])
    # print(parsed_line[14])
    # print(parsed_line[15])
    #
    # i=0
    # while i<4:
    #
    #     if parsed_line[2+i] == '0':
    #         if parsed_line[7+i] == '1':
    #             FP_count+=1
    #         elif parsed_line[7+i] == '0':
    #             TN_count+=1
    #
    #     elif parsed_line[2+i] == '1':
    #         if parsed_line[7+i] == '1':
    #             if float(parsed_line[12+i])>=threshold_value:
    #                 TP_count+=1
    #             else:
    #                 FP_count+=1
    #         elif parsed_line[7+i] == '0':
    #             FN_count+=1
    #     else:
    #         print('First line does not have correct indication!')
    #     i+=1


    for line in f:
    # line=f.readline()

        parsed_line=line.split()
        file_name_array=parsed_line[0].split('/')
        file_name=file_name_array[3]
        folder_name=file_name_array[2]


        # print(file_name_array[9])
        # print(folder_fields[0].index(file_name_array[9]))

        i=0
        while i<4:

            if parsed_line[2+i] == '0':
                if parsed_line[7+i] == '1':
                    FP_count+=1
                    folder_fields[2][folder_fields[0].index(file_name_array[9])]+=1
                elif parsed_line[7+i] == '0':
                    TN_count+=1
                    folder_fields[3][folder_fields[0].index(file_name_array[9])]+=1
            elif parsed_line[2+i] == '1':
                if parsed_line[7+i] == '1':
                    if float(parsed_line[12+i])>=threshold_value:
                        TP_count+=1
                        folder_fields[1][folder_fields[0].index(file_name_array[9])]+=1

                    else:
                        FP_count+=1
                        folder_fields[2][folder_fields[0].index(file_name_array[9])]+=1

                elif parsed_line[7+i] == '0':
                    FN_count+=1
                    folder_fields[4][folder_fields[0].index(file_name_array[9])]+=1

            else:
                print('Line does not have correct indication!')
            i+=1


    print('TP count:'+str(TP_count))
    print('TN count:'+str(TN_count))
    print('FP count:'+str(FP_count))
    print('FN count:'+str(FN_count))
    beta=1
    Precision=(TP_count)/(TP_count+FP_count)
    Recall=(TP_count)/(TP_count+FN_count)
    print('Precision=TP/(TP+FP)='+str(Precision))
    print('Recall=TP/(TP+FN)='+str(Recall))
    F_measure=(1+int(beta)*int(beta))*((Precision*Recall)/(Precision+Recall))
    F_measure2=(2*Precision*Recall)/(Precision+Recall)
    print('F-measure=(1+beta^2)*(Precision*Recall)/(beta^2*Precision+Recall)='+str(F_measure2))
    print('F-measure='+str(F_measure2*100)+'%')
    print('')
    for i,element in enumerate(folder_fields[0]):
        print('')
        print(element)
        print('TP: '+str(folder_fields[1][i]))
        print('FP: '+str(folder_fields[2][i]))
        print('TN: '+str(folder_fields[3][i]))
        print('FN: '+str(folder_fields[4][i]))
        Precision=(folder_fields[1][i])/(folder_fields[1][i]+folder_fields[2][i])
        Recall=(folder_fields[1][i])/(folder_fields[1][i]+folder_fields[4][i])
        print('Precision=TP/(TP+FP)='+str(Precision))
        print('Recall=TP/(TP+FN)='+str(Recall))
        F_measure=(1+int(beta)*int(beta))*((Precision*Recall)/(Precision+Recall))
        F_measure2=(2*Precision*Recall)/(Precision+Recall)
        print('F-measure=(1+beta^2)*(Precision*Recall)/(beta^2*Precision+Recall)='+str(F_measure2))
        print('F-measure='+str(F_measure2*100)+'%')


    f.close()

    return

###################################################################################################
if __name__ == "__main__":
    main()
