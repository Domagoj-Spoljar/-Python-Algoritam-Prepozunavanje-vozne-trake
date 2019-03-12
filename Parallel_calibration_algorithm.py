import numpy as np
import cv2
import os
import sys
import function_parameters as FP
import copy
import pickle

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

framesss=0
filename = 'frame_count'
filename2 = 'algorithm_status'
filename3 = 'calculated_binary_combinations'


def calibrate_loop():
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

        calibration_function_new.main(str(new_count))
        FP.binary_combinations=(FP.calibrated_combinations[0][0][1],FP.calibrated_combinations[0][1][1],FP.calibrated_combinations[1][0][1])
        print('???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????')
        print(FP.binary_combinations)

        outfile3 = open(filename3,'wb')
        pickle.dump(FP.binary_combinations,outfile3)
        outfile3.close()
        # calibration_function_new.main(str(FP.calibration_frame))
    # print('FP.frame= '+str(FP.frame))
def calibrate_loop_once():

    calibration_function_new.main(str(0))
    FP.binary_combinations=(FP.calibrated_combinations[0][0][1],FP.calibrated_combinations[0][1][1],FP.calibrated_combinations[1][0][1])
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

if __name__ == '__main__':
    outfile2 = open(filename2,'wb')
    value=True
    pickle.dump(value,outfile2)
    outfile2.close()

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

    service.close()
    worker.close()

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
