import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import Image_processing_functions as IPF
import sys
import function_parameters as FP

def main():
    if len(sys.argv) == 2:
        img_arg=sys.argv[1]
        print('Processing image: '+img_arg)
    else:
        img_arg='frame'+str(FP.frame)
        print('Processing default image: '+img_arg)

    dashcam_image_path = FP.dashcam_image_path
    exampleImg = cv2.imread(dashcam_image_path+img_arg+".jpg")
    exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)

    exampleImg_undistort = IPF.undistort(exampleImg)
    h,w = exampleImg_undistort.shape[:2]
    # define source and destination points for transform
    src,dst = FP.unwarp_points(h,w)

    exampleImg_unwarp, M, Minv = IPF.unwarp(exampleImg_undistort, src, dst)
    img_unwarp=exampleImg_unwarp
    img_unwarp_inverted=cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2BGR)




    # Visualize multiple color space channels
    exampleImg_unwarp_R = exampleImg_unwarp[:,:,0]
    exampleImg_unwarp_G = exampleImg_unwarp[:,:,1]
    exampleImg_unwarp_B = exampleImg_unwarp[:,:,2]
    exampleImg_unwarp_HSV = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2HSV)
    exampleImg_unwarp_H = exampleImg_unwarp_HSV[:,:,0]
    exampleImg_unwarp_S = exampleImg_unwarp_HSV[:,:,1]
    exampleImg_unwarp_V = exampleImg_unwarp_HSV[:,:,2]
    exampleImg_unwarp_LAB = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2Lab)
    exampleImg_unwarp_L = exampleImg_unwarp_LAB[:,:,0]
    exampleImg_unwarp_A = exampleImg_unwarp_LAB[:,:,1]
    exampleImg_unwarp_B2 = exampleImg_unwarp_LAB[:,:,2]

    #_______________________________________________________________________________________________________________________________
    # Visualize undistortion


    f, ax = plt.subplots(4, 3, figsize=(20,10))
    f.subplots_adjust(hspace = .1, wspace=0.01)

    ax[0,0].imshow(exampleImg_undistort)
    ax[0,0].axis('off')
    ax[0,0].set_title('Undistorted Image', fontsize=15)

    ax[0,1].imshow(exampleImg_undistort)
    x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
    y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
    ax[0,1].plot(x, y, color='#ff0000', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
    ax[0,1].axis('off')
    ax[0,1].set_title('Undistorted Image with ROI', fontsize=15)

    ax[0,2].imshow(exampleImg_unwarp)
    ax[0,2].axis('off')
    ax[0,2].set_title('Unwarped Image', fontsize=15)

    #----------------------------------------------------------------
    ax[1,0].imshow(exampleImg_unwarp_R, cmap='gray')
    ax[1,0].set_title('RGB R-channel', fontsize=15)
    ax[1,0].axis('off')
    ax[1,1].imshow(exampleImg_unwarp_G, cmap='gray')
    ax[1,1].set_title('RGB G-Channel', fontsize=15)
    ax[1,1].axis('off')
    ax[1,2].imshow(exampleImg_unwarp_B, cmap='gray')
    ax[1,2].set_title('RGB B-channel', fontsize=15)
    ax[1,2].axis('off')
    ax[2,0].imshow(exampleImg_unwarp_H, cmap='gray')
    ax[2,0].set_title('HSV H-Channel', fontsize=15)
    ax[2,0].axis('off')
    ax[2,1].imshow(exampleImg_unwarp_S, cmap='gray')
    ax[2,1].set_title('HSV S-channel', fontsize=15)
    ax[2,1].axis('off')
    ax[2,2].imshow(exampleImg_unwarp_V, cmap='gray')
    ax[2,2].set_title('HSV V-Channel', fontsize=15)
    ax[2,2].axis('off')
    ax[3,0].imshow(exampleImg_unwarp_L, cmap='gray')
    ax[3,0].set_title('LAB L-channel', fontsize=15)
    ax[3,0].axis('off')
    ax[3,1].imshow(exampleImg_unwarp_A, cmap='gray')
    ax[3,1].set_title('LAB A-Channel', fontsize=15)
    ax[3,1].axis('off')
    ax[3,2].imshow(exampleImg_unwarp_B2, cmap='gray')
    ax[3,2].set_title('LAB B-Channel', fontsize=15)
    ax[3,2].axis('off')


    #-----------------------------------------------------------
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    #-----------------------------------------------------------


    lista_white=['rgb_r','hls_s','hls_l','lab_l','hsv_white','white_tight','white_loose']
    stacked_binary_image_white,all_binary_images_white=IPF.make_binary_stack_custom(img_unwarp,lista_white)
    max_value_white=np.max(stacked_binary_image_white)
    threshold_white=IPF.threshold_equation(max_value_white)
    thresholded_binary_image_white=IPF.threshold_binary_stack(stacked_binary_image_white,threshold_white)


    ffx, axy = plt.subplots(3, 4, figsize=(20,10))
    ffx.subplots_adjust(hspace = .1, wspace=0.01)

    #-------------------------------------------------------------
    axy[0,0].imshow(all_binary_images_white[0],cmap='gray')
    axy[0,0].axis('off')
    axy[0,0].set_title('rgb_r', fontsize=12)

    #------------------------------------------------------------------

    axy[0,1].imshow(all_binary_images_white[1],cmap='gray')
    axy[0,1].axis('off')
    axy[0,1].set_title('hls_s', fontsize=12)

    #------------------------------------------------------------------
    axy[0,2].imshow(all_binary_images_white[2],cmap='gray')
    axy[0,2].axis('off')
    axy[0,2].set_title('hls_l', fontsize=12)

    #-------------------------------------------------------------------
    axy[0,3].imshow(all_binary_images_white[3],cmap='gray')
    axy[0,3].axis('off')
    axy[0,3].set_title('lab_l', fontsize=12)
    #-------------------------------------------------------------------

    axy[1,0].imshow(all_binary_images_white[4],cmap='gray')
    axy[1,0].axis('off')
    axy[1,0].set_title('hsv white', fontsize=15)

    #-------------------------------------------------------------------
    axy[1,1].imshow(all_binary_images_white[5],cmap='gray')
    axy[1,1].axis('off')
    axy[1,1].set_title('white_tight', fontsize=12)
    # # #-------------------------------------------------------------------
    axy[1,2].imshow(all_binary_images_white[6],cmap='gray')
    axy[1,2].axis('off')
    axy[1,2].set_title('white_loose', fontsize=12)

    axy[2,0].imshow(stacked_binary_image_white,cmap='gray')
    axy[2,0].axis('off')
    axy[2,0].set_title('stacked all together', fontsize=12)


    axy[2,1].imshow(thresholded_binary_image_white,cmap='gray')
    axy[2,1].axis('off')
    axy[2,1].set_title('thresholded: '+str(threshold_white), fontsize=12)

    lista_sobel=['sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel','yellow_edge_pos','yellow_edge_neg']
    stacked_binary_image_sobel,all_binary_images_sobel=IPF.make_binary_stack_custom(img_unwarp,lista_sobel)
    stacked_binary_image_white=stacked_binary_image_white+stacked_binary_image_sobel
    max_value_white=np.max(stacked_binary_image_white)
    threshold_white=IPF.threshold_equation(max_value_white)
    thresholded_binary_image_white=IPF.threshold_binary_stack(stacked_binary_image_white,threshold_white)

    axy[2,2].imshow(stacked_binary_image_white,cmap='gray')
    axy[2,2].axis('off')
    axy[2,2].set_title('stacked all together with sobel ', fontsize=12)

    axy[2,3].imshow(thresholded_binary_image_white,cmap='gray')
    axy[2,3].axis('off')
    axy[2,3].set_title('thresholded (sobel): '+str(threshold_white), fontsize=12)






# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
    lista_yellow=['lab_b','hsv_yellow','yellow','yellow_2','yellow_3','yellow_4','yellow_5']
    stacked_binary_image_yellow,all_binary_images_yellow=IPF.make_binary_stack_custom(img_unwarp_inverted,lista_yellow)
    max_value_yellow=np.max(stacked_binary_image_yellow)
    threshold_yellow=IPF.threshold_equation(max_value_yellow)
    thresholded_binary_image_yellow=IPF.threshold_binary_stack(stacked_binary_image_yellow,threshold_yellow)

    ff, axx = plt.subplots(3, 4, figsize=(20,10))
    ff.subplots_adjust(hspace = .1, wspace=0.01)



    axx[0,0].imshow(all_binary_images_yellow[0],cmap='gray')
    axx[0,0].axis('off')
    axx[0,0].set_title('lab_b', fontsize=12)
    # #------------------------------------------------------------------
    axx[0,1].imshow(all_binary_images_yellow[1],cmap='gray')
    axx[0,1].axis('off')
    axx[0,1].set_title('hsv_yellow', fontsize=12)
    #----------------------------------------------------------
    axx[0,2].imshow(all_binary_images_yellow[2],cmap='gray')
    axx[0,2].axis('off')
    axx[0,2].set_title('yellow', fontsize=12)

    axx[0,3].imshow(all_binary_images_yellow[3],cmap='gray')
    axx[0,3].axis('off')
    axx[0,3].set_title('yellow_2', fontsize=12)

    axx[1,0].imshow(all_binary_images_yellow[4],cmap='gray')
    axx[1,0].axis('off')
    axx[1,0].set_title('yellow_3', fontsize=12)

    axx[1,1].imshow(all_binary_images_yellow[5],cmap='gray')
    axx[1,1].axis('off')
    axx[1,1].set_title('yellow_4', fontsize=12)

    axx[1,2].imshow(all_binary_images_yellow[6],cmap='gray')
    axx[1,2].axis('off')
    axx[1,2].set_title('yellow_5', fontsize=12)

    axx[2,0].imshow(stacked_binary_image_yellow,cmap='gray')
    axx[2,0].axis('off')
    axx[2,0].set_title('stacked all together', fontsize=12)

    axx[2,1].imshow(thresholded_binary_image_yellow,cmap='gray')
    axx[2,1].axis('off')
    axx[2,1].set_title('thresholded: '+str(threshold_yellow), fontsize=12)

    stacked_binary_image_yellow=stacked_binary_image_yellow+stacked_binary_image_sobel
    max_value_yellow=np.max(stacked_binary_image_yellow)
    threshold_yellow=IPF.threshold_equation(max_value_yellow)
    thresholded_binary_image_yellow=IPF.threshold_binary_stack(stacked_binary_image_yellow,threshold_yellow)


    axx[2,2].imshow(stacked_binary_image_yellow,cmap='gray')
    axx[2,2].axis('off')
    axx[2,2].set_title('stacked all together with sobel ', fontsize=12)

    axx[2,3].imshow(thresholded_binary_image_yellow,cmap='gray')
    axx[2,3].axis('off')
    axx[2,3].set_title('thresholded (sobel): '+str(threshold_yellow), fontsize=12)


    # #------------------------------------------------------
    ff, bxx = plt.subplots(4, 3, figsize=(20,10))
    ff.subplots_adjust(hspace = .1, wspace=0.01)

    # lista_sobel=['sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel','yellow_edge_pos','yellow_edge_neg']

    bxx[0,0].imshow(all_binary_images_sobel[0],cmap='gray')
    bxx[0,0].axis('off')
    bxx[0,0].set_title('sobel_mag', fontsize=15)
    #-------------------------------------------------------------------
    bxx[0,1].imshow(all_binary_images_sobel[1],cmap='gray')
    bxx[0,1].axis('off')
    bxx[0,1].set_title('sobel_abs', fontsize=15)
    #--------------------------------------------------------------

    bxx[0,2].imshow(all_binary_images_sobel[2],cmap='gray')
    bxx[0,2].axis('off')
    bxx[0,2].set_title('sobel_dir', fontsize=15)


    bxx[1,0].imshow(all_binary_images_sobel[3],cmap='gray')
    bxx[1,0].axis('off')
    bxx[1,0].set_title('edge_pos', fontsize=12)
    #-------------------------------------------------------------------
    bxx[1,1].imshow(all_binary_images_sobel[4],cmap='gray')
    bxx[1,1].axis('off')
    bxx[1,1].set_title('edge_neg', fontsize=12)
    #-------------------------------------------------------------------
    bxx[1,2].imshow(all_binary_images_sobel[5],cmap='gray')
    bxx[1,2].axis('off')
    bxx[1,2].set_title('hls_sobel', fontsize=12)
    #-------------------------------------------------------------------
    bxx[2,0].imshow(all_binary_images_sobel[6],cmap='gray')
    bxx[2,0].axis('off')
    bxx[2,0].set_title('yellow_edge_pos', fontsize=12)

    bxx[2,1].imshow(all_binary_images_sobel[7],cmap='gray')
    bxx[2,1].axis('off')
    bxx[2,1].set_title('yellow_edge_neg', fontsize=12)

    lista_all=['rgb_r','hls_s','hls_l','lab_l','hsv_white','white_tight','white_loose','lab_b','hsv_yellow','yellow','yellow_2','yellow_3','yellow_4','yellow_5','sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel']
    stacked_binary_image_all,all_binary_images=IPF.make_binary_stack_custom(img_unwarp,lista_all)
    max_value=np.max(stacked_binary_image_all)
    threshold_value=IPF.threshold_equation(max_value)
    thresholded_binary_image=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value)
    threshold_value_custom=5
    thresholded_binary_image_custom=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom)

    bxx[2,2].imshow(stacked_binary_image_all,cmap='gray')
    bxx[2,2].axis('off')
    bxx[2,2].set_title('all filters stacked', fontsize=12)


    bxx[3,2].imshow(thresholded_binary_image,cmap='gray')
    bxx[3,2].axis('off')
    bxx[3,2].set_title('all filters thresholded: '+str(threshold_value), fontsize=12)


    bxx[3,1].imshow(thresholded_binary_image_custom,cmap='gray')
    bxx[3,1].axis('off')
    bxx[3,1].set_title('all filters thresholded: '+str(threshold_value_custom), fontsize=12)



    sharpen_img=np.copy(exampleImg_unwarp)

    gausian_img=cv2.GaussianBlur(sharpen_img,(5,5),0)
    sharpened=cv2.addWeighted(exampleImg_unwarp,1.5,gausian_img,-0.5,0)
    bxx[3,0].imshow(sharpened,cmap='gray')
    bxx[3,0].axis('off')
    bxx[3,0].set_title('sharpened ', fontsize=12)








    ff, cxx = plt.subplots(4, 3, figsize=(20,10))
    ff.subplots_adjust(hspace = .1, wspace=0.01)

    lista_all=['rgb_r','hls_s','hls_l','lab_l','hsv_white','white_tight','white_loose','lab_b','hsv_yellow','yellow','yellow_2','yellow_3','yellow_4','yellow_5','sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel']
    stacked_binary_image_all,all_binary_images=IPF.make_binary_stack_custom(img_unwarp,lista_all)
    max_value=np.max(stacked_binary_image_all)
    threshold_value=IPF.threshold_equation(max_value)
    thresholded_binary_image=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value)



    cxx[0,0].imshow(stacked_binary_image_all,cmap='gray')
    cxx[0,0].axis('off')
    cxx[0,0].set_title('all filters stacked', fontsize=12)

    cxx[3,2].imshow(thresholded_binary_image,cmap='gray')
    cxx[3,2].axis('off')
    cxx[3,2].set_title('all filters AUTO thresholded: '+str(threshold_value), fontsize=12)

    threshold_value_custom1=1
    thresholded_binary_image_custom1=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom1)
    threshold_value_custom2=2
    thresholded_binary_image_custom2=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom2)
    threshold_value_custom3=3
    thresholded_binary_image_custom3=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom3)
    threshold_value_custom4=4
    thresholded_binary_image_custom4=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom4)
    threshold_value_custom5=5
    thresholded_binary_image_custom5=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom5)
    threshold_value_custom6=6
    thresholded_binary_image_custom6=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom6)
    threshold_value_custom7=7
    thresholded_binary_image_custom7=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom7)
    threshold_value_custom8=8
    thresholded_binary_image_custom8=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom8)
    threshold_value_custom9=9
    thresholded_binary_image_custom9=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom9)
    threshold_value_custom10=10
    thresholded_binary_image_custom10=IPF.threshold_binary_stack(stacked_binary_image_all,threshold_value_custom10)


    cxx[0,1].imshow(thresholded_binary_image_custom1,cmap='gray')
    cxx[0,1].axis('off')
    cxx[0,1].set_title('all filters thresholded: '+str(threshold_value_custom1), fontsize=12)

    cxx[0,2].imshow(thresholded_binary_image_custom2,cmap='gray')
    cxx[0,2].axis('off')
    cxx[0,2].set_title('all filters thresholded: '+str(threshold_value_custom2), fontsize=12)

    cxx[1,0].imshow(thresholded_binary_image_custom3,cmap='gray')
    cxx[1,0].axis('off')
    cxx[1,0].set_title('all filters thresholded: '+str(threshold_value_custom3), fontsize=12)

    cxx[1,1].imshow(thresholded_binary_image_custom4,cmap='gray')
    cxx[1,1].axis('off')
    cxx[1,1].set_title('all filters thresholded: '+str(threshold_value_custom4), fontsize=12)

    cxx[1,2].imshow(thresholded_binary_image_custom5,cmap='gray')
    cxx[1,2].axis('off')
    cxx[1,2].set_title('all filters thresholded: '+str(threshold_value_custom5), fontsize=12)

    cxx[2,0].imshow(thresholded_binary_image_custom6,cmap='gray')
    cxx[2,0].axis('off')
    cxx[2,0].set_title('all filters thresholded: '+str(threshold_value_custom6), fontsize=12)

    cxx[2,1].imshow(thresholded_binary_image_custom7,cmap='gray')
    cxx[2,1].axis('off')
    cxx[2,1].set_title('all filters thresholded: '+str(threshold_value_custom7), fontsize=12)

    cxx[2,2].imshow(thresholded_binary_image_custom8,cmap='gray')
    cxx[2,2].axis('off')
    cxx[2,2].set_title('all filters thresholded: '+str(threshold_value_custom8), fontsize=12)

    cxx[3,0].imshow(thresholded_binary_image_custom9,cmap='gray')
    cxx[3,0].axis('off')
    cxx[3,0].set_title('all filters thresholded: '+str(threshold_value_custom9), fontsize=12)

    cxx[3,1].imshow(thresholded_binary_image_custom10,cmap='gray')
    cxx[3,1].axis('off')
    cxx[3,1].set_title('all filters thresholded: '+str(threshold_value_custom10), fontsize=12)
    #
    # cxx[0,1].imshow(thresh_binary_image,cmap='gray')
    # cxx[0,1].axis('off')
    # cxx[0,1].set_title('SOBEL with calc THRESHOLD: '+str(threshold_value), fontsize=12)
    #
    #
    # cxx[1,0].imshow(stacked_binary_image_white,cmap='gray')
    # cxx[1,0].axis('off')
    # cxx[1,0].set_title('WHITE added all together', fontsize=12)
    #
    # # combined_white_sobel=np.zeros_like(thresh_binary_image)
    # # combined_white_sobel+=thresh_binary_image
    # # combined_white_sobel+=stacked_binary_image_white
    # stacked_binary_white_sobel=thresh_binary_image+stacked_binary_image_white
    #
    # cxx[1,1].imshow(stacked_binary_white_sobel,cmap='gray')
    # cxx[1,1].axis('off')
    # cxx[1,1].set_title('WHITE + SOBEL added', fontsize=12)
    #
    # max_value2=np.max(stacked_binary_white_sobel)
    # threshold_value2=IPF.threshold_equation(max_value2)
    # stacked_thresholded_white_sobel=IPF.threshold_binary_stack(stacked_binary_white_sobel,threshold_value2)
    # cxx[1,2].imshow(stacked_thresholded_white_sobel,cmap='gray')
    # cxx[1,2].axis('off')
    # cxx[1,2].set_title('WHITE + SOBEL calc thresholded: '+str(threshold_value2), fontsize=12)
    #
    # cxx[2,0].imshow(stacked_binary_image_yellow,cmap='gray')
    # cxx[2,0].axis('off')
    # cxx[2,0].set_title('YELLOW added all together', fontsize=12)
    #
    #
    # stacked_binary_yellow_sobel=thresh_binary_image+stacked_binary_image_yellow
    # cxx[2,1].imshow(stacked_binary_yellow_sobel,cmap='gray')
    # cxx[2,1].axis('off')
    # cxx[2,1].set_title('YELLOW + SOBEL added', fontsize=12)
    #
    # max_value3=np.max(stacked_binary_image_yellow)
    # threshold_value3=IPF.threshold_equation(max_value3)
    # stacked_thresholded_yellow_sobel=IPF.threshold_binary_stack(stacked_binary_image_yellow,threshold_value3)
    # cxx[2,2].imshow(stacked_thresholded_yellow_sobel,cmap='gray')
    # cxx[2,2].axis('off')
    # cxx[2,2].set_title('YELLOW + SOBEL  calc thresholded: '+str(threshold_value3), fontsize=12)

    #------------------------------------------------------
    plt.show()

    return

###################################################################################################
if __name__ == "__main__":
    main()
