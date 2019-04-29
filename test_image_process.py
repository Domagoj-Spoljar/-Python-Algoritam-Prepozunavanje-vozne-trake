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
import text_print_functions as TPF

count=0

def main():
    global count
    if len(sys.argv) == 2:
        count=sys.argv[1]
        print('Processing image: frame'+str(count))
    else:
        # count=str(FP.frame)
        # count=str(FP.calibration_frame)
        print('Processing default image: frame'+str(count))

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
    print("_______________________________________________________________")
    print('Calibrating image filters',end="", flush=True)
    lista_white=['rgb_r','hls_s','hls_l','lab_l','hsv_white','white_tight','white_loose']
    lista_yellow=['lab_b','hsv_yellow','yellow','yellow_2','yellow_3','yellow_4','yellow_5']
    stacked_binary_image_white,all_binary_images_white=IPF.make_binary_stack_custom(img_unwarp,lista_white)
    stacked_binary_image_yellow,all_binary_images_yellow=IPF.make_binary_stack_custom(img_unwarp,lista_yellow)


    cv2.imwrite('[make_binary_stack_custom(1)]rgb_r.png',all_binary_images_white[0].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(2)]hls_s.png',all_binary_images_white[1].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(3)]hls_l.png',all_binary_images_white[2].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(4)]lab_l.png',all_binary_images_white[3].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(5)]hsv_white.png',all_binary_images_white[4].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(6)]white_tight.png',all_binary_images_white[5].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(7)]white_loose.png',all_binary_images_white[6].astype('uint8') * 255)


    cv2.imwrite('[make_binary_stack_custom(11)]lab_b.png',all_binary_images_yellow[0].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(22)]hsv_yellow.png',all_binary_images_yellow[1].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(33)]yellow.png',all_binary_images_yellow[2].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(44)]yellow_2.png',all_binary_images_yellow[3].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(55)]yellow_3.png',all_binary_images_yellow[4].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(66)]yellow_4.png',all_binary_images_yellow[5].astype('uint8') * 255)
    cv2.imwrite('[make_binary_stack_custom(77)]yellow_5.png',all_binary_images_yellow[6].astype('uint8') * 255)




    sobel=True
    if sobel:
        lista_sobel=['sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel','yellow_edge_pos','yellow_edge_neg']
        stacked_binary_image_sobel,all_binary_images_sobel=IPF.make_binary_stack_custom(img_unwarp,lista_sobel)
        stacked_binary_image_white=stacked_binary_image_white+stacked_binary_image_sobel
        stacked_binary_image_yellow=stacked_binary_image_yellow+stacked_binary_image_sobel

        cv2.imwrite('[make_binary_stack_custom(111)]sobel_mag.png',all_binary_images_sobel[0].astype('uint8') * 255)
        cv2.imwrite('[make_binary_stack_custom(222)]sobel_abs.png',all_binary_images_sobel[1].astype('uint8') * 255)
        cv2.imwrite('[make_binary_stack_custom(333)]sobel_dir.png',all_binary_images_sobel[2].astype('uint8') * 255)
        cv2.imwrite('[make_binary_stack_custom(444)]edge_pos.png',all_binary_images_sobel[3].astype('uint8') * 255)
        cv2.imwrite('[make_binary_stack_custom(555)]edge_neg.png',all_binary_images_sobel[4].astype('uint8') * 255)
        cv2.imwrite('[make_binary_stack_custom(666)]hls_sobel.png',all_binary_images_sobel[5].astype('uint8') * 255)
        cv2.imwrite('[make_binary_stack_custom(777)]yellow_edge_pos.png',all_binary_images_sobel[6].astype('uint8') * 255)
        cv2.imwrite('[make_binary_stack_custom(888)]yellow_edge_neg.png',all_binary_images_sobel[7].astype('uint8') * 255)


    print('.',end="", flush=True)
    max_value_white=np.max(stacked_binary_image_white)
    max_value_yellow=np.max(stacked_binary_image_yellow)

    threshold_white=IPF.threshold_equation(max_value_white)
    threshold_yellow=IPF.threshold_equation(max_value_yellow)
    # print(threshold_yellow)
    # print(threshold_white)
    print('.',end="", flush=True)

    thresholded_binary_image_white=IPF.threshold_binary_stack(stacked_binary_image_white,threshold_white)
    thresholded_binary_image_yellow=IPF.threshold_binary_stack(stacked_binary_image_yellow,threshold_yellow)

    print('.')
    top_yellow=IPF.compare_binary_images_NEW(thresholded_binary_image_yellow,all_binary_images_yellow,lista_yellow,diagnostic=True)
    print('')
    top_white=IPF.compare_binary_images_NEW(thresholded_binary_image_white,all_binary_images_white,lista_white,diagnostic=True)
    print('+'+'_'*(TPF.line_length-2)+'+')
    print(TPF.print_line_text_in_middle('Done!',TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')
    # print('Done!')
    print(TPF.print_line_in_defined_length('Best combinations are (yellow)',TPF.line_length-2))
    print(TPF.print_line_3_columns(top_yellow[0][1],top_yellow[1][1]  , top_yellow[2][1] ,TPF.line_length-2))
    print('| '+'-'*(TPF.line_length-4)+' |')

    print(TPF.print_line_in_defined_length('Best combinations are (white)',TPF.line_length-2))
    print(TPF.print_line_3_columns(top_white[0][1],top_white[1][1]  , top_white[2][1] ,TPF.line_length-2))
    print('+'+'_'*(TPF.line_length-2)+'+')

    # image_HSV = cv2.cvtColor(img_unwarp,cv2.COLOR_BGR2HSV)
    # yellow_hsv_low2  = np.array([ 0, 80, 200])
    # yellow_hsv_high2 = np.array([ 40, 255, 255])
    # res_mask2 = cv2.inRange(image_HSV,yellow_hsv_low2,yellow_hsv_high2)
    # res_mask2[(res_mask2 > 0)] = 1
    # cv2.imwrite('[make_binary_stack_custom(44)]yellow_22.png',res_mask2.astype('uint8') * 255)


#--------------------------------------------------------------------------------------------------------------------


###################################################################################################
if __name__ == "__main__":
    main()
