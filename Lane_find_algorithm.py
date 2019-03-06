import numpy as np
import cv2
import os
import sys
import function_parameters as FP

import OLD_calibrate_ipf
import Process_image
import Test_with_function
import frames_to_video
import Histogram_peaks
import Process_image_new
import Complete_algorithm

video_option=0

def main():
    while True:
        print('')
        print('***Lane Find Algorithm***')
        print('Author: Domagoj Špoljar')
        print(' ')
        print('Options:')
        print('MAIN')
        print('1. Find (Calibrate) best parameters for image processing')
        print('2. Process image (frame) - NEW')
        print('3. Run complete algrithm on one frame')
        print('4. Create video')
        print('')
        print('ADDITIONAL')
        print('5. Make histogram of frame image')
        print('6. Process image (frame) - OLD')
        print('')
        print('9. Run complete algorithm (Calibration,create video)')
        print('')
        print('0. EXIT')
        print('')
        option1 = input("What do you want to do? ")

        while str(option1)!='0' and str(option1)!='1' and str(option1)!='2' and str(option1)!='3' and str(option1)!='4'and str(option1)!='5'and str(option1)!='6'and str(option1)!='9':
            print('Wrong input entered. Please try again!')
            option1 = input("What do you want to do? ")
        print('')

        if str(option1)=='0':
            break

        print('Available videos:')
        print('1. challenge_video.mp4 (Sunny, w/o trespassing)')
        print('2. project_video.mp4 (Sunny, w/o trespassing)')
        print('3. test_video.mp4 (Sunny, w/ trespassing)')
        print('4. harder_challenge_video.mp4')
        print('5. night_video.mp4')
        print('6. foggy_video')
        print('7. rainy_video')
        print('8. rainy_video_better')
        print('9. rainy_video2')
        print('10. test_video.mp4 (640x360)')
        print('11. project_video.mp4 (640x360)')
        option2 = input("Choose video: ")
        print('')

        while str(option2)!='1' and str(option2)!='2' and str(option2)!='3' and str(option2)!='4'and str(option2)!='5'and str(option2)!='6'and str(option2)!='7'and str(option2)!='8'and str(option2)!='9'and str(option2)!='10'and str(option2)!='11':
            print('Wrong input entered. Please try again!')
            option2 = input("Choose video: ")
        print('')

        if str(option2)=='1':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/challnege_video/'
            FP.video_tip= 'challenge video'
        elif str(option2)=='2':
            FP.dashcam_image_path = '/home/profesor/Documents/Datasets/project_video/'
            FP.video_tip= 'project_video'
        elif str(option2)=='3':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/dashcam_driving/'
            FP.video_tip= 'test_video'
        elif str(option2)=='4':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/harder_challenge_video/'
            FP.video_tip= 'harder_challenge_video'
        elif str(option2)=='5':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/night/'
            FP.video_tip= 'night_video'
        elif str(option2)=='6':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/foggy_video/'
            FP.video_tip= 'foggy_video'
        elif str(option2)=='7':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/rainy_video/'
            FP.video_tip= 'rainy_video'
        elif str(option2)=='8':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/rainy_video_better/'
            FP.video_tip= 'rainy_video_better'
        elif str(option2)=='9':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/rainy_video2/'
            FP.video_tip= 'rainy_video2'
        elif str(option2)=='10':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/dashcam_driving-640x360/'
            FP.video_tip= 'dashcam_driving_640x360'
        elif str(option2)=='11':
            FP.dashcam_image_path='/home/profesor/Documents/Datasets/project_video640x360/'
            FP.video_tip= 'project_video_640x360'



        if str(option1)=='3' or str(option1)=='4'or str(option1)=='5':
            print('Choose binary combinations:')
            print('1. RGB-r')
            print('2. HLS-s')
            print('3. HLS-l')
            print('4. LAB-l')
            print('5. LAB-b')
            print('6. sobel_mag')
            print('7. sobel_abs')
            print('8. sobel_dir')
            print('9. hsv_white')
            print('10. hsv_yellow')
            print('11. white_thight')
            print('12. white_loose')
            print('13. yellow_edge_pos')
            print('14. yellow_edge_neg')
            print('15. yellow')
            print('16. edge_pos')
            print('17. edge_neg')
            print('18. hls_sobel')
            print('0. DONE')
            print('')

            if len(FP.calibrated_combinations)>0 and option2==video_option:
                print('Calibrated combinations (white): '+FP.calibrated_combinations[0][0][1]+', '+FP.calibrated_combinations[0][1][1])
                print('Calibrated combinations (yellow): '+FP.calibrated_combinations[1][0][1]+', '+FP.calibrated_combinations[1][1][1])
                print('')

            combinations=[]
            rez_binary_combinations=()

            option7 = input("Choose combination: ")

            while str(option7)!='0':
                while str(option7)!='1' and str(option7)!='2' and str(option7)!='3' and str(option7)!='4'and str(option7)!='5'and str(option7)!='6'and str(option7)!='7'and str(option7)!='8'and str(option7)!='9'and str(option7)!='10'and str(option7)!='11'and str(option7)!='12'and str(option7)!='13'and str(option7)!='14'and str(option7)!='15'and str(option7)!='16'and str(option7)!='17'and str(option7)!='18'and str(option7)!='0':
                    print('Wrong input entered. Please try again!')
                    option7 = input("Choose combination: ")
                while option7 in combinations:
                    print('Combination allready entered!')
                    option7 = input("Please enter another one: ")
                combinations.append(option7)
                option7 = input("Choose another combination: ")

            print('')

            if(len(combinations)!=0):
                for x in combinations:
                    if str(x)=='1':
                        rez_binary_combinations+=('rgb_r',)
                    if str(x)=='2':
                        rez_binary_combinations+=('hls_s',)
                    if str(x)=='3':
                        rez_binary_combinations+=('hls_l',)
                    if str(x)=='4':
                        rez_binary_combinations+=('lab_l',)
                    if str(x)=='5':
                        rez_binary_combinations+=('lab_b',)
                    if str(x)=='6':
                        rez_binary_combinations+=('sobel_mag',)
                    if str(x)=='7':
                        rez_binary_combinations+=('sobel_abs',)
                    if str(x)=='8':
                        rez_binary_combinations+=('sobel_dir',)
                    if str(x)=='9':
                        rez_binary_combinations+=('hsv_white',)
                    if str(x)=='10':
                        rez_binary_combinations+=('hsv_yellow',)
                    if str(x)=='11':
                        rez_binary_combinations+=('white_tight',)
                    if str(x)=='12':
                        rez_binary_combinations+=('white_loose',)
                    if str(x)=='13':
                        rez_binary_combinations+=('yellow_edge_pos',)
                    if str(x)=='14':
                        rez_binary_combinations+=('yellow_edge_neg',)
                    if str(x)=='15':
                        rez_binary_combinations+=('yellow',)
                    if str(x)=='16':
                        rez_binary_combinations+=('edge_pos',)
                    if str(x)=='17':
                        rez_binary_combinations+=('edge_neg',)
                    if str(x)=='18':
                        rez_binary_combinations+=('hls_sobel',)
                FP.binary_combinations=rez_binary_combinations

            else:
                print('No combinations entered. Processing with default combinations '+str(FP.binary_combinations))

            print('You choose: '+str(rez_binary_combinations))

        if str(option1)=='1' or str(option1)=='2' or str(option1)=='3'or str(option1)=='5'or str(option1)=='6':
            print('Available frames:')
            print('1. Default frame ('+str(FP.frame)+')')
            print('2. Custom frame')
            option3 = input("Choose frame: ")
            print('')

            while str(option3)!='1' and str(option3)!='2':
                print('Wrong input entered. Please try again!')
                option3 = input("Choose frame: ")
            print('')

            if str(option3)=='2':
                option4 = input("Enter custom frame:")
                FP.frame=str(option4)

        elif str(option1)=='4' or str(option1)=='9':

            print('Choose video type:')
            print('1. Diagnostic')
            print('2. Fullscreen')
            option5 = input("Choose: ")
            print('')
            while str(option5)!='1' and str(option5)!='2':
                print('Wrong input entered. Please try again!')
                option5 = input("Choose: ")
            if str(option5)=='1':
                FP.video_type='diagnostic'
                FP.fullscreen=False
            elif str(option5)=='2':
                FP.video_type='fullscreen'
                FP.fullscreen=True


            FP.video_name =FP.video_tip+'_'+FP.video_type+'_'+FP.version +'.mp4'
            print('')
            print('Choose file (video) name:')
            print('1. Default: '+ str(FP.video_name))
            print('2. Custom')

            option3 = input("Choose: ")
            print('')
            while str(option3)!='1' and str(option3)!='2':
                print('Wrong input entered. Please try again!')
                option3 = input("Choose: ")
            print('')
            if str(option3)=='1':

                option4 = input("Enter video version (current: "+ FP.version+"):")
                FP.version=str(option4)
                FP.video_name =FP.video_tip+'_'+FP.video_type +'_'+FP.version +'.mp4'
            elif str(option3)=='2':
                option4 = input("Enter custom video name:")
                FP.video_name=str(option4)+'.mp4'
            print('')

        video_option=option2

        if str(option1)=='1':
            OLD_calibrate_ipf.main()
            # FP.binary_combinations=lista
        elif str(option1)=='2':
            Process_image_new.main()
        elif str(option1)=='3':
            Test_with_function.main()
        elif str(option1)=='4':
            frames_to_video.main()
        elif str(option1)=='5':
            Histogram_peaks.main()
        elif str(option1)=='6':
            Process_image.main()


        elif str(option1)=='9':
            OLD_calibrate_ipf.main()
            FP.binary_combinations=(FP.calibrated_combinations[0][0][1],FP.calibrated_combinations[0][1][1],FP.calibrated_combinations[1][0][1])
            # FP.binary_combinations=(FP.calibrated_combinations[0][0][1],FP.calibrated_combinations[0][1][1])
            frames_to_video.main()
    return

###################################################################################################
if __name__ == "__main__":
    main()
