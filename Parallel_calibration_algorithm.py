import numpy as np
import cv2
import os
import sys
import function_parameters as FP
import copy
import pickle
# import operator

import OLD_calibrate_ipf
import Process_image
import Test_with_function
import frames_to_video
import frames_to_video_dynamic
import Histogram_peaks
import Process_image_new
import Complete_algorithm
import calibration_function_new
import Test_with_function_noKeys
import multiprocessing as mp
from multiprocessing import Process, Queue
import text_print_functions as TPF

# def info(title):
#     print(title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())
#
# def f(name):
#     info('function f')
#     print('hello', name)

# frame_number=3

# frame_number
# frame_number=FP.calibration_frame
# frame_number=FP.frame

# calibration_leaderboard =	{
# 'rgb_r':0,
# 'hls_s':0,
# 'hls_l':0,
# 'lab_l':1,
# 'hsv_white':0,
# 'white_tight':0,
# 'white_loose':0,
# 'lab_b':0,
# 'hsv_yellow':0,
# 'yellow':0
# }
# [count,calib_simmilar_percent,delta_change_percent]
# calibration_leaderboard_white_scores=[[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]

# total_count,count_memory,percent_memory,weights,final_score
calibration_leaderboard_white =	['rgb_r','hls_s','hls_l','lab_l','hsv_white','white_tight','white_loose']
calibration_leaderboard_white_scores=[[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]]]

calibration_leaderboard_yellow =['lab_b','hsv_yellow','yellow','yellow_2','yellow_3','yellow_4','yellow_5']
calibration_leaderboard_yellow_scores =[[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]],
[0,[0,0,0,0,0],[0,0,0,0,0],[0.1,0.1,0.1,0.1,0.1],0,[0,0,0,0,0]]]


image_width=1280
image_height=720
total_pixel_num=image_width*image_height


framesss=0
filename = 'frame_count'
filename2 = 'algorithm_status'
filename3 = 'calculated_binary_combinations'

def update_leaderboards(white_filters,yellow_filters):
    global calibration_leaderboard_white,calibration_leaderboard_white_scores,total_pixel_num
    # for white filters
    name=white_filters[0][1]
    number=calibration_leaderboard_white.index(name)
    calibration_leaderboard_white_scores[number][0]+=1


    print('calibration_leaderboard_white start')
    # print(calibration_leaderboard_white_scores)
    for i, elements in enumerate(white_filters):
        name=white_filters[i][1]
        print('name= '+ str(name))
        number=calibration_leaderboard_white.index(name)
        print('number= '+ str(number))
        # calibration_leaderboard_white_previous=np.copy(calibration_leaderboard_white_scores[number][1])
        percent=((((white_filters[i][0]/total_pixel_num)*100)+100)/200)*100
        # calibration_leaderboard_white_scores[number][1]=((((white_filters[i][0]/total_pixel_num)*100)+100)/200)*100
        # calibration_leaderboard_white_scores[number][1]=(white_filters[i][0]/total_pixel_num)*100
        # calibration_leaderboard_white_scores[number][2]+=(calibration_leaderboard_white_scores[number][1]/calibration_leaderboard_white_previous)-1

        calibration_leaderboard_white_scores[number][1].insert(0,percent)
        del calibration_leaderboard_white_scores[number][1][-1]


        calibration_leaderboard_white_scores[number][2].insert(0,calibration_leaderboard_white_scores[number][0])
        del calibration_leaderboard_white_scores[number][2][-1]

        calibration_leaderboard_white_scores[number][3].insert(0,(0.1*calibration_leaderboard_white_scores[number][0]+0.1))
        del calibration_leaderboard_white_scores[number][3][-1]

        a=np.array(calibration_leaderboard_white_scores[i][1])
        b=np.array(calibration_leaderboard_white_scores[i][3])
        calibration_leaderboard_white_scores[i][5]=a*b
        calibration_leaderboard_white_scores[i][4]=np.sum(calibration_leaderboard_white_scores[i][5])

    print('+'+'_'*(TPF.line_length-2)+'+')
    print(TPF.print_line_text_in_middle('Leaderboard white',TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')
    print(TPF.print_line_3_columns('count','% with ref image', 'stored %' ,TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')
    length=len(calibration_leaderboard_white)
    for i in range(length):
    # for i, elements in enumerate(calibration_leaderboard_white_scores):
        print(TPF.print_line_in_defined_length(calibration_leaderboard_white[i],TPF.line_length-2))
        print(TPF.print_line_in_defined_length("count",TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_white_scores[i][2]),TPF.line_length-2))
        print(TPF.print_line_in_defined_length("stored %",TPF.line_length-2))
        # print(TPF.print_line_3_columns(str(calibration_leaderboard_white_scores[i][0]),'-',"-",TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_white_scores[i][1]),TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_white_scores[i][3]) ,TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_white_scores[i][5]) ,TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_white_scores[i][4]) ,TPF.line_length-2))

        print('| '+'-'*(TPF.line_length-4)+' |')


    print('+'+'_'*(TPF.line_length-2)+'+')
    print('calibration_leaderboard_white end')
    print(calibration_leaderboard_white)
    print('calibration_leaderboard_white_score')
    print(calibration_leaderboard_white_scores)
    print('MAX_INDEX')
    total_calc=[]
    print(len(calibration_leaderboard_white_scores))
    total_length=len(calibration_leaderboard_white_scores)
    for i in range(total_length):
        total_calc.append((calibration_leaderboard_white[i],calibration_leaderboard_white_scores[i][4]))
    print('total_calc')
    print(total_calc)

    max_value=max(total_calc,key=lambda item:item[1])
    print('max_value')
    print(max_value)
    print(max_value[0])
# ------------------------------------------------------------------
    # for yellow filters
    name=yellow_filters[0][1]
    number=calibration_leaderboard_yellow.index(name)
    calibration_leaderboard_yellow_scores[number][0]+=1

    print('calibration_leaderboard_yellow start')
    # print(calibration_leaderboard_white_scores)
    for i, elements in enumerate(yellow_filters):
        name=yellow_filters[i][1]
        print('name= '+ str(name))
        number=calibration_leaderboard_yellow.index(name)
        print('number= '+ str(number))
        # calibration_leaderboard_white_previous=np.copy(calibration_leaderboard_white_scores[number][1])
        # percent=((((yellow_filters[i][0]/total_pixel_num)*100)+100)/200)*100
        percent=((((yellow_filters[i][0]/total_pixel_num)*100)+100)/200)*1000
        # calibration_leaderboard_white_scores[number][1]=((((white_filters[i][0]/total_pixel_num)*100)+100)/200)*100
        # calibration_leaderboard_white_scores[number][1]=(white_filters[i][0]/total_pixel_num)*100
        # calibration_leaderboard_white_scores[number][2]+=(calibration_leaderboard_white_scores[number][1]/calibration_leaderboard_white_previous)-1

        calibration_leaderboard_yellow_scores[number][1].insert(0,percent)
        del calibration_leaderboard_yellow_scores[number][1][-1]


        calibration_leaderboard_yellow_scores[number][2].insert(0,calibration_leaderboard_yellow_scores[number][0])
        del calibration_leaderboard_yellow_scores[number][2][-1]

        calibration_leaderboard_yellow_scores[number][3].insert(0,(0.1*calibration_leaderboard_yellow_scores[number][0]+0.1))
        del calibration_leaderboard_yellow_scores[number][3][-1]

        a=np.array(calibration_leaderboard_yellow_scores[i][1])
        b=np.array(calibration_leaderboard_yellow_scores[i][3])
        calibration_leaderboard_yellow_scores[i][5]=a*b
        calibration_leaderboard_yellow_scores[i][4]=np.sum(calibration_leaderboard_yellow_scores[i][5])

    print('+'+'_'*(TPF.line_length-2)+'+')
    print(TPF.print_line_text_in_middle('Leaderboard yellow',TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')
    print(TPF.print_line_3_columns('count','% with ref image', 'stored %' ,TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')
    length=len(calibration_leaderboard_yellow)
    for i in range(length):
    # for i, elements in enumerate(calibration_leaderboard_white_scores):
        print(TPF.print_line_in_defined_length(calibration_leaderboard_yellow[i],TPF.line_length-2))
        print(TPF.print_line_in_defined_length("count",TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_yellow_scores[i][2]),TPF.line_length-2))
        print(TPF.print_line_in_defined_length("stored %",TPF.line_length-2))
        # print(TPF.print_line_3_columns(str(calibration_leaderboard_white_scores[i][0]),'-',"-",TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_yellow_scores[i][1]),TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_yellow_scores[i][3]) ,TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_yellow_scores[i][5]) ,TPF.line_length-2))
        print(TPF.print_line_in_defined_length(str(calibration_leaderboard_yellow_scores[i][4]) ,TPF.line_length-2))

        print('| '+'-'*(TPF.line_length-4)+' |')


    print('+'+'_'*(TPF.line_length-2)+'+')
    print('calibration_leaderboard_yellow end')
    print(calibration_leaderboard_yellow)
    print('calibration_leaderboard_yellow_score')
    print(calibration_leaderboard_yellow_scores)
    print('MAX_INDEX')
    total_calc=[]
    print(len(calibration_leaderboard_yellow_scores))
    total_length=len(calibration_leaderboard_yellow_scores)
    for i in range(total_length):
        total_calc.append((calibration_leaderboard_yellow[i],calibration_leaderboard_yellow_scores[i][4]))
    print('total_calc')
    print(total_calc)

    max_value2=max(total_calc,key=lambda item:item[1])
    print('max_value')
    print(max_value2)
    print(max_value2[0])

    if max_value2[1]>9:
        final_list=(max_value[0],max_value2[0])
    else:
        final_list=(max_value[0])

    return final_list




def update_leaderboards_first_time(white_filters,yellow_filters):
    global calibration_leaderboard_white,calibration_leaderboard_white_scores,total_pixel_num

    name=white_filters[0][1]
    number=calibration_leaderboard_white.index(name)
    calibration_leaderboard_white_scores[number][0]+=1

    print('calibration_leaderboard_white start')
    for i, elements in enumerate(white_filters):
        name=white_filters[i][1]
        print('name= '+ str(name))
        number=calibration_leaderboard_white.index(name)
        print('number= '+ str(number))
        # calibration_leaderboard_white_previous=np.copy(calibration_leaderboard_white_scores[number][1])
        calibration_leaderboard_white_scores[number][1]=((((white_filters[i][0]/total_pixel_num)*100)+100)/200)*100
        # calibration_leaderboard_white_scores[number][1]=(white_filters[i][0]/total_pixel_num)*100
        # calibration_leaderboard_white_scores[number][2]=calibration_leaderboard_white_scores[number][1]/calibration_leaderboard_white_previous


    print('+'+'_'*(TPF.line_length-2)+'+')
    print(TPF.print_line_text_in_middle('Leaderboard',TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')
    print(TPF.print_line_3_columns('count','% with ref image', '% with prev calibration' ,TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')
    length=len(calibration_leaderboard_white)
    for i in range(length):
    # for i, elements in enumerate(calibration_leaderboard_white_scores):
        print(TPF.print_line_in_defined_length(calibration_leaderboard_white[i],TPF.line_length-2))
        print(TPF.print_line_3_columns(str(calibration_leaderboard_white_scores[i][0]),str(calibration_leaderboard_white_scores[i][1])+'%',str(calibration_leaderboard_white_scores[i][2])+'%' ,TPF.line_length-2))
        print('| '+'-'*(TPF.line_length-4)+' |')
        # print(i)

    print('+'+'_'*(TPF.line_length-2)+'+')

    print('calibration_leaderboard_white end')
    print(calibration_leaderboard_white)
    print('calibration_leaderboard_white_score')
    print(calibration_leaderboard_white_scores)

    final_list=('rgb_r','lab_b')
    return final_list


def calibrate_loop():
    global calibration_leaderboard_white,total_pixel_num
    # global algotirthm_running
    while True:
        # global framesss
        # global frame_number
        # FP.calibration_frame=frame_number
        # print('%%%%%%%%%%%%%%%%%%%%%%%%CALIBRATION_frame_number= '+str(frame_number)+'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # frame=check_frame()
        # print('algotirthm_running is '+str(algotirthm_running))

        infile2 = open(filename2,'rb')
        end_of_algorithm = pickle.load(infile2)
        infile2.close()
        # print('end_of_algorithm= '+str(end_of_algorithm))
        if end_of_algorithm is False:
            # print('FFalse')
            break

        infile = open(filename,'rb')
        new_count = pickle.load(infile)
        infile.close()

        white_filters,yellow_filters=calibration_function_new.main(str(new_count))

        # print('calibration_leaderboard_white')
        # for i in len(white_filters):
        #     name=white_filters[i][1]
        #     # number=calibration_leaderboard_white[:][0][0].index(name)
        #     # calibration_leaderboard_white[number][1][0]=white_filters[i][0]/total_pixel_num
        #     # calibration_leaderboard_white[number][1][1]=
        #     # calibration_leaderboard_white[number][1][2]+=1
        # print(calibration_leaderboard_white)

        final_filters=update_leaderboards(white_filters,yellow_filters)

        # FP.binary_combinations=('rgb_r')
        FP.binary_combinations=final_filters
        # FP.binary_combinations=white_filters[0][1],white_filters[1][1],yellow_filters[0][1]
        # FP.binary_combinations=(FP.calibrated_combinations[0][0][1],FP.calibrated_combinations[0][1][1],FP.calibrated_combinations[1][0][1])

        print('???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????')
        print(FP.binary_combinations)

        outfile3 = open(filename3,'wb')
        pickle.dump(FP.binary_combinations,outfile3)
        outfile3.close()
        # calibration_function_new.main(str(FP.calibration_frame))
    # print('FP.frame= '+str(FP.frame))
def calibrate_loop_once():

    white_filters,yellow_filters=calibration_function_new.main(str(0))
    final_filters=update_leaderboards(white_filters,yellow_filters)
    FP.binary_combinations=final_filters
    # FP.binary_combinations=(white_filters[0][1],white_filters[1][1],yellow_filters[0][1])
    outfile3 = open(filename3,'wb')
    pickle.dump(FP.binary_combinations,outfile3)
    outfile3.close()



        # print(new_count)
        # print('new_count= '+str(new_count))

# def calibrate_loop():
#     while True:
#         # global frame_number
#         # FP.calibration_frame=frame_number
#         # print('%%%%%%%%%%%%%%%%%%%%%%%%CALIBRATION_frame_number= '+str(frame_number)+'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#         frame=check_frame()
#         calibration_function_new.main(str(frame))
#         # calibration_function_new.main(str(FP.calibration_frame))

def algorithm_function():
    # global frame_number
    # while True:
    # Test_with_function_noKeys.main()
    frames_to_video_dynamic.main()
    # global algotirthm_running
    outfile2 = open(filename2,'wb')
    value=False
    pickle.dump(value,outfile2)
    outfile2.close()
    # print('algotirthm_running is '+str(algotirthm_running))
    # print('FP.frame= '+str(FP.frame))
    # framesss=copy.deepcopy(FP.frame)
    # print('framesss= '+str(framesss))
    # frame_number+=1
    # print('%%%%%%%%%%%%%%%%%%%%%%%%frame_number= '+str(FP.calibration_frame)+'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

# def calibrate_loop():
#     while True:
#         global frame_number
#         FP.calibration_frame=frame_number
#         # print('%%%%%%%%%%%%%%%%%%%%%%%%CALIBRATION_frame_number= '+str(frame_number)+'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#         calibration_function_new.main()
#
# def algorithm_function():
#     global frame_number
#     # while True:
#     frames_to_video.main()
#     frame_number=FP.frame
#     print('%%%%%%%%%%%%%%%%%%%%%%%%frame_number= '+str(frame_number)+'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


def main():
    outfile2 = open(filename2,'wb')
    value=True
    pickle.dump(value,outfile2)
    outfile2.close()

    outfile1 = open(filename,'wb')
    value=0
    pickle.dump(value,outfile1)
    outfile1.close()



    calibrate_loop_once()
    # slika = multiprocessing.Value('i')
    # global frame_number
    # frame_number=FP.frame
    # algorithm_function()
    # service=Process(target=calibrate_loop, args=(frame_number,))
    service=Process(target=calibrate_loop)
    worker=Process(target=algorithm_function)
    # worker=Process(target=f, args=('2222',))

    service.start()
    worker.start()

    service.join()
    worker.join()

    # service.close()
    # worker.close()

if __name__ == '__main__':
    main()


    # infile = open(filename,'rb')
    # new_count = pickle.load(infile)
    # infile.close()
    # print(new_count)


# if __name__ == '__main__':
#     # mp.set_start_method('fork')
#     # service=Process(target=Test_with_function.main())
#     # q = Queue()
#     service=Process(target=OLD_calibrate_ipf.main())
#     FP.binary_combinations=(FP.calibrated_combinations[0][0][1],FP.calibrated_combinations[0][1][1],FP.calibrated_combinations[1][0][1])
#     # q.put(FP.calibrated_combinations)
#     worker = Process(target=Test_with_function.main())
#     count=Test_with_function.count
#     print('count is: '+str(count))
#     if
#     # q.put(FP.frame)
#     service.start()
#     worker.start()
#
#     # print(q.get())
#
#     service.join()
#     worker.join()
