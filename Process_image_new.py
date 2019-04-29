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

    min_thresh=25
    max_thresh=255
    exampleImg_sobelAbs = IPF.sobel_abs_thresh(exampleImg_unwarp, 'x', min_thresh, max_thresh)

    min_thresh2=25
    max_thresh2=255
    kernel_size=25
    exampleImg_sobelMag = IPF.sobel_mag_thresh(exampleImg_unwarp, kernel_size, (min_thresh2, max_thresh2))

    min_thresh3=0
    max_thresh3=0.11
    kernel_size2=7
    exampleImg_sobelDir = IPF.sobel_dir_thresh(exampleImg_unwarp, kernel_size2, (min_thresh3, max_thresh3))

    sobelMag_sobelDir = np.zeros_like(exampleImg_sobelMag)
    sobelMag_sobelDir[((exampleImg_sobelMag == 1) & (exampleImg_sobelDir == 1))] = 1

    sobelAbs_sobelDir = np.zeros_like(exampleImg_sobelAbs)
    sobelAbs_sobelDir[((exampleImg_sobelAbs == 1) & (exampleImg_sobelDir == 1))] = 1

    sobelMag_sobelAbs = np.zeros_like(exampleImg_sobelMag)
    sobelMag_sobelAbs[((exampleImg_sobelMag == 1) & (exampleImg_sobelAbs == 1))] = 1

    exampleImg_SThresh = IPF.hls_threshold(exampleImg_unwarp,thresh=(125,255),color='s')
    exampleImg_LThresh = IPF.hls_threshold(exampleImg_unwarp,thresh=(220,255),color='l')
    # exampleImg_LBThresh = IPF.lab_threshold(img_unwarp_inverted,thresh=(150,255),color='b')
    exampleImg_LBThresh = IPF.lab_threshold(img_unwarp_inverted,thresh=(190,255),color='b')
    exampleImg_LLBThresh = IPF.lab_threshold(exampleImg_unwarp,thresh=(190,255),color='l')
    exampleImg_RRGBThresh = IPF.rgb_thresh(exampleImg_unwarp,color=0)
    exampleImg_GRGBThresh = IPF.rgb_thresh(exampleImg_unwarp,color=1)
    exampleImg_BRGBThresh = IPF.rgb_thresh(exampleImg_unwarp,color=2)


    #delete plot
    #added_binary_images=np.zeros_like(exampleImg_unwarp)
    added_binary_images=np.zeros_like(exampleImg_unwarp[:,:,0])
    #print(added_binary_images.dtype)
    added_binary_images=sobelMag_sobelAbs+sobelAbs_sobelDir+sobelMag_sobelDir+exampleImg_BRGBThresh+exampleImg_GRGBThresh+exampleImg_RRGBThresh+exampleImg_LLBThresh+exampleImg_LBThresh+exampleImg_LThresh+exampleImg_SThresh


    # thresh_binary_images=np.zeros_like(exampleImg_unwarp[:,:,0])
    # thresh_binary_images[(added_binary_images[:,:] > 3)] = 1

    #thresh_binary_images3=IPF.binary_threshold(added_binary_images,3)
    thresh_binary_image,_=IPF.make_binary_stack(exampleImg_unwarp)

    thresh_binary_images2=IPF.threshold_binary_stack(thresh_binary_image,2)
    thresh_binary_images3=IPF.threshold_binary_stack(thresh_binary_image,3)
    thresh_binary_images4=IPF.threshold_binary_stack(thresh_binary_image,4)
    thresh_binary_images5=IPF.threshold_binary_stack(thresh_binary_image,5)
    thresh_binary_images6=IPF.threshold_binary_stack(thresh_binary_image,6)
    thresh_binary_images7=IPF.threshold_binary_stack(thresh_binary_image,7)
    # thresh_binary_images4=IPF.binary_threshold(added_binary_images,4)
    # thresh_binary_images5=IPF.binary_threshold(added_binary_images,5)
    # thresh_binary_images6=IPF.binary_threshold(added_binary_images,6)
    # thresh_binary_images7=IPF.binary_threshold(added_binary_images,7)

    # max_value=np.max(thresh_binary_image)
    #
    # slike=[exampleImg_RRGBThresh,exampleImg_GRGBThresh,exampleImg_BRGBThresh,sobelMag_sobelDir,exampleImg_LThresh,exampleImg_SThresh,sobelMag_sobelAbs,sobelAbs_sobelDir,exampleImg_LBThresh,exampleImg_LLBThresh]
    # razlika=IPF.compare_binary_images(thresh_binary_images2,slike)
    # print('max value is '+str(max_value))


    # first,second=IPF.calibrate_IPF(exampleImg_unwarp)

    kanali = IPF.split_channels(exampleImg_unwarp)
    #_______________________________________________________________________________________________________________________________
    def color_mask(hsv,low,high):
    # Return mask from HSV
        mask = cv2.inRange(hsv, low, high)
        return mask

    def apply_color_mask(hsv,img,low,high):
        # Apply color mask to image
        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_and(img,img, mask= mask)
        return res

    image_HSV = cv2.cvtColor(exampleImg_unwarp,cv2.COLOR_RGB2HSV)
    yellow_hsv_low  = np.array([ 0,  100,  100])
    yellow_hsv_high = np.array([ 80, 255, 255])
    white_hsv_low  = np.array([ 0,   0,   160])
    white_hsv_high = np.array([ 255,  80, 255])
    res_mask = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    res_mask[(res_mask > 0)] = 1
    # res1 = apply_color_mask(image_HSV,exampleImg_unwarp,white_hsv_low,white_hsv_high)
    res1 = color_mask(image_HSV,white_hsv_low,white_hsv_high)


    yellow_hsv_low2  = np.array([ 0, 80, 200])
    yellow_hsv_high2 = np.array([ 40, 255, 255])
    res_mask2 = cv2.inRange(image_HSV,yellow_hsv_low2,yellow_hsv_high2)
    res_mask2[(res_mask2 > 0)] = 1

    yellow_hsv_low3  = np.array([ 15, 38, 115])
    yellow_hsv_high3 = np.array([ 35, 204, 255])
    res_mask3 = cv2.inRange(image_HSV,yellow_hsv_low3,yellow_hsv_high3)
    res_mask3[(res_mask3 > 0)] = 1

    yellow_hsv_low4  = np.array([ 20, 120, 80])
    yellow_hsv_high4 = np.array([ 45, 200, 255])
    res_mask4 = cv2.inRange(image_HSV,yellow_hsv_low4,yellow_hsv_high4)
    res_mask4[(res_mask4 > 0)] = 1

    yellow_hsv_low5  = np.array([ 0, 100, 100])
    yellow_hsv_high5 = np.array([ 50, 255, 255])
    res_mask5 = cv2.inRange(image_HSV,yellow_hsv_low5,yellow_hsv_high5)
    res_mask5[(res_mask5 > 0)] = 1





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

    # ax[1,3].imshow(exampleImg_sobelAbs,cmap='gray')
    # ax[1,3].axis('off')
    # ax[1,3].set_title('Sobel Absolute', fontsize=15)
    # #--------------------------------------------------------------
    #
    # ax[2,3].imshow(exampleImg_sobelMag,cmap='gray')
    # ax[2,3].axis('off')
    # ax[2,3].set_title('Sobel Magnitude', fontsize=15)
    # #--------------------------------------------------------------
    #
    # ax[3,3].imshow(exampleImg_sobelDir,cmap='gray')
    # ax[3,3].axis('off')
    # ax[3,3].set_title('Sobel Direction', fontsize=15)


    #_______________________________________________________________________________________________________________________________
    #_______________________________________________________________________________________________________________________________
    #--------------------------------------------------------------

    ffx, axy = plt.subplots(3, 4, figsize=(20,10))
    ffx.subplots_adjust(hspace = .1, wspace=0.01)

    #-------------------------------------------------------------
    axy[0,1].imshow(exampleImg_SThresh,cmap='gray')
    axy[0,1].axis('off')
    axy[0,1].set_title('HLS S-channel thresholded', fontsize=12)

    #------------------------------------------------------------------

    axy[2,2].imshow(exampleImg_LThresh,cmap='gray')
    axy[2,2].axis('off')
    axy[2,2].set_title('HLS L-channel thresholded', fontsize=12)

    #------------------------------------------------------------------


    #------------------------------------------------------------------
    axy[1,1].imshow(exampleImg_LLBThresh,cmap='gray')
    axy[1,1].axis('off')
    axy[1,1].set_title('LAB L-channel thresholded', fontsize=12)

    #------------------------------------------------------------------

    # axy[0,3].imshow(sobelMag_sobelDir,cmap='gray')
    # axy[0,3].axis('off')
    # axy[0,3].set_title('Sobel Direction+Sobel magnitude', fontsize=12)
    #-------------------------------------------------------------------
    axy[0,0].imshow(exampleImg_RRGBThresh,cmap='gray')
    axy[0,0].axis('off')
    axy[0,0].set_title('RGB r thresholded', fontsize=12)
    #-------------------------------------------------------------------
    # axy[0,1].imshow(exampleImg_GRGBThresh,cmap='gray')
    # axy[0,1].axis('off')
    # axy[0,1].set_title('RGB g thresholded', fontsize=12)
    # #-------------------------------------------------------------------
    # axy[0,2].imshow(exampleImg_BRGBThresh,cmap='gray')
    # axy[0,2].axis('off')
    # axy[0,2].set_title('RGB b thresholded', fontsize=12)


    # axy[1,0].imshow(exampleImg_sobelDir,cmap='gray')
    # axy[1,0].axis('off')
    # axy[1,0].set_title('Sobel Direction', fontsize=15)
    # #-------------------------------------------------------------------
    # axy[2,0].imshow(sobelMag_sobelAbs,cmap='gray')
    # axy[2,0].axis('off')
    # axy[2,0].set_title('Sobel mag+abs', fontsize=12)
    # #-------------------------------------------------------------------
    # axy[2,2].imshow(sobelAbs_sobelDir,cmap='gray')
    # axy[2,2].axis('off')
    # axy[2,2].set_title('Sobel abs+dir', fontsize=12)
    # #-------------------------------------------------------------------
    #
    # axy[1,3].imshow(exampleImg_sobelAbs,cmap='gray')
    # axy[1,3].axis('off')
    # axy[1,3].set_title('Sobel Absolute', fontsize=15)
    # #--------------------------------------------------------------
    #
    # axy[1,2].imshow(exampleImg_sobelMag,cmap='gray')
    # axy[1,2].axis('off')
    # axy[1,2].set_title('Sobel Magnitude', fontsize=15)
    #--------------------------------------------------------------

    axy[1,0].imshow(res1,cmap='gray')
    axy[1,0].axis('off')
    axy[1,0].set_title('hsv white', fontsize=15)

    # axy[3,0].imshow(kanali['edge_pos'],cmap='gray')
    # axy[3,0].axis('off')
    # axy[3,0].set_title('edge_pos', fontsize=12)
    #
    # #------------------------------------------------------------------
    # # _, sxbinary = threshold(hls_hls[:, :, 1], thresh=(120, 255))
    # axy[3,1].imshow(kanali['edge_neg'],cmap='gray')
    # axy[3,1].axis('off')
    # axy[3,1].set_title('edge_neg', fontsize=12)

    axy[2,1].imshow(kanali['white_tight'],cmap='gray')
    axy[2,1].axis('off')
    axy[2,1].set_title('white_tight', fontsize=12)
    # # #-------------------------------------------------------------------
    axy[2,0].imshow(kanali['white_loose'],cmap='gray')
    axy[2,0].axis('off')
    axy[2,0].set_title('white_loose', fontsize=12)

    s_thresh=(150, 255)
    l_thresh=(120,255)
    sx_thresh=(20, 100)
    hls = cv2.cvtColor(exampleImg_unwarp, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Apply Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # This will take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from the horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Apply Thresholding
    final_binary = np.zeros_like(s_channel)
    final_binary[np.logical_or((s_channel > s_thresh[0]) & (s_channel < s_thresh[1]) & (l_channel > l_thresh[0]) & (l_channel < l_thresh[1]) , (scaled_sobel > sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))] = 1

    lista_white=['rgb_r','hls_s','hls_l','lab_l','hsv_white','white_tight','white_loose']
    stacked_binary_image_white,all_binary_images_white=IPF.make_binary_stack_custom(exampleImg_unwarp,lista_white)
    max_value_white=np.max(stacked_binary_image_white)

    threshold_white=IPF.threshold_equation(max_value_white)
    thresholded_binary_image_white=IPF.threshold_binary_stack(stacked_binary_image_white,threshold_white)

    axy[0,2].imshow(stacked_binary_image_white,cmap='gray')
    axy[0,2].axis('off')
    axy[0,2].set_title('stacked all together', fontsize=12)


    axy[0,3].imshow(thresholded_binary_image_white,cmap='gray')
    axy[0,3].axis('off')
    axy[0,3].set_title('thresholded: '+str(threshold_white), fontsize=12)

    lista_sobel=['sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel']
    stacked_binary_image_sobel,all_binary_images_sobel=IPF.make_binary_stack_custom(exampleImg_unwarp,lista_sobel)
    stacked_binary_image_sobel=stacked_binary_image_white+stacked_binary_image_sobel
    max_value_sobel=np.max(stacked_binary_image_sobel)
    threshold_sobel=IPF.threshold_equation(max_value_sobel)
    thresholded_binary_image_sobel=IPF.threshold_binary_stack(stacked_binary_image_sobel,threshold_sobel)

    # stacked_binary_image_combined =np.zeros_like(stacked_binary_image_sobel)
    # stacked_binary_image_combined=stacked_binary_image_white+stacked_binary_image_sobel
    # thresholded_binary_image_sobel=IPF.threshold_binary_stack(stacked_binary_image_combined,threshold_sobel)

    # thresholded_binary_image_combined =np.zeros_like(thresholded_binary_image_sobel)
    # thresholded_binary_image_combined =np.zeros_like(thresholded_binary_image_sobel)
    # thresholded_binary_image_combined[(thresholded_binary_image_sobel==1)&(thresholded_binary_image_white==1)]=1
    axy[1,2].imshow(stacked_binary_image_sobel,cmap='gray')
    axy[1,2].axis('off')
    axy[1,2].set_title('stacked all together with sobel ', fontsize=12)
    axy[1,3].imshow(thresholded_binary_image_sobel,cmap='gray')
    axy[1,3].axis('off')
    axy[1,3].set_title('thresholded (sobel): '+str(threshold_sobel), fontsize=12)
# --------------------------------------------------------------------------------------------------------------------------
    ff, axx = plt.subplots(4, 4, figsize=(20,10))
    ff.subplots_adjust(hspace = .1, wspace=0.01)

    # axx[0,0].imshow(kanali['edge_pos'],cmap='gray')0    # axx[0,0].axis('off')
    # axx[0,0].set_title('edge_pos', fontsize=12)
    #
    # #------------------------------------------------------------------
    # # _, sxbinary = threshold(hls_hls[:, :, 1], thresh=(120, 255))
    # axx[0,1].imshow(kanali['edge_neg'],cmap='gray'1
    # axx[0,1].axis('off')
    # axx[0,1].set_title('edge_neg', fontsize=12)

    # #------------------------------------------------------------------
    # sxbinary = blur_gaussian(sxbinary, ksize=3)
    axx[0,0].imshow(kanali['yellow_edge_pos'],cmap='gray')
    axx[0,0].axis('off')
    axx[0,0].set_title('yellow_edge_pos', fontsize=12)
    # #------------------------------------------------------------------
    axx[0,1].imshow(kanali['yellow_edge_neg'],cmap='gray')
    axx[0,1].axis('off')
    axx[0,1].set_title('yellow_edge_neg', fontsize=12)
    #----------------------------------------------------------
    axx[1,0].imshow(kanali['yellow'],cmap='gray')
    axx[1,0].axis('off')
    axx[1,0].set_title('yellow', fontsize=12)





    axx[1,1].imshow(exampleImg_LBThresh,cmap='gray')
    axx[1,1].axis('off')
    axx[1,1].set_title('LAB B-channel thresholded', fontsize=12)

    axx[2,0].imshow(res_mask,cmap='gray')
    axx[2,0].axis('off')
    axx[2,0].set_title('HSV yellow', fontsize=12)

    axx[1,2].imshow(res_mask2,cmap='gray')
    axx[1,2].axis('off')
    axx[1,2].set_title('HSV yellow 2', fontsize=12)

    axx[1,3].imshow(res_mask3,cmap='gray')
    axx[1,3].axis('off')
    axx[1,3].set_title('HSV yellow 3', fontsize=12)

    axx[3,0].imshow(res_mask4,cmap='gray')
    axx[3,0].axis('off')
    axx[3,0].set_title('HSV yellow 4', fontsize=12)

    axx[3,1].imshow(res_mask5,cmap='gray')
    axx[3,1].axis('off')
    axx[3,1].set_title('HSV yellow 5', fontsize=12)



    lista_yellow=['lab_b','hsv_yellow','yellow_edge_pos','yellow_edge_neg','yellow','yellow_2','yellow_3','yellow_4','yellow_5']
    stacked_binary_image_yellow,all_binary_images_yellow=IPF.make_binary_stack_custom(exampleImg_unwarp,lista_yellow)
    max_value_yellow=np.max(stacked_binary_image_yellow)

    threshold_yellow=IPF.threshold_equation(max_value_yellow)
    thresholded_binary_image_yellow=IPF.threshold_binary_stack(stacked_binary_image_yellow,threshold_yellow)
    axx[2,2].imshow(stacked_binary_image_yellow,cmap='gray')
    axx[2,2].axis('off')
    axx[2,2].set_title('combined', fontsize=12)
    axx[2,3].imshow(thresholded_binary_image_yellow,cmap='gray')
    axx[2,3].axis('off')
    axx[2,3].set_title('combined and thresholded '+str(threshold_yellow), fontsize=12)



    lista_sobel=['sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel']
    stacked_binary_image_sobel,all_binary_images_sobel=IPF.make_binary_stack_custom(exampleImg_unwarp,lista_sobel)
    stacked_binary_image_sobel=stacked_binary_image_yellow+stacked_binary_image_sobel
    max_value_sobel=np.max(stacked_binary_image_sobel)
    threshold_sobel=IPF.threshold_equation(max_value_sobel)
    thresholded_binary_image_sobel=IPF.threshold_binary_stack(stacked_binary_image_sobel,threshold_sobel)

    axx[0,2].imshow(stacked_binary_image_sobel,cmap='gray')
    axx[0,2].axis('off')
    axx[0,2].set_title('stacked all together with sobel ', fontsize=12)
    axx[0,3].imshow(thresholded_binary_image_sobel,cmap='gray')
    axx[0,3].axis('off')
    axx[0,3].set_title('thresholded (sobel): '+str(threshold_sobel), fontsize=12)


    def get_sobel_bin(img):
        ''' "img" should be 1-channel '''

        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)  # x-direction gradient
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        sobel_bin = np.zeros_like(scaled_sobel)
        sobel_bin[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        return sobel_bin

    def get_threshold(img):
        ''' "img" should be an undistorted image '''

        # Color-space conversions
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # Sobel gradient binaries
        sobel_s_bin = get_sobel_bin(s_channel)
        sobel_gray_bin = get_sobel_bin(gray)

        sobel_comb_bin = np.zeros_like(sobel_s_bin)
        sobel_comb_bin[(sobel_s_bin == 1) | (sobel_gray_bin == 1)] = 1

        # HLS S-Channel binary
        s_bin = np.zeros_like(s_channel)
        s_bin[(s_channel >= 150) & (s_channel <= 255)] = 1

        # Combine the binaries
        comb_bin = np.zeros_like(sobel_comb_bin)
        comb_bin[(sobel_comb_bin == 1) | (s_bin == 1)] = 1

        gray_img = np.dstack((gray, gray, gray))
        sobel_s_img = np.dstack((sobel_s_bin, sobel_s_bin, sobel_s_bin))*255
        sobel_gray_img = np.dstack((sobel_gray_bin, sobel_gray_bin, sobel_gray_bin))*255
        sobel_comb_img = np.dstack((sobel_comb_bin, sobel_comb_bin, sobel_comb_bin))*255
        s_img = np.dstack((s_bin, s_bin, s_bin))*255
        comb_img = np.dstack((comb_bin, comb_bin, comb_bin))*255

        return comb_img

    threshold_img = get_threshold(exampleImg_unwarp)
    axx[2,1].imshow(threshold_img,cmap='gray')
    axx[2,1].axis('off')
    axx[2,1].set_title('sobel neki threshold', fontsize=12)




    # axx[2,1].imshow(res1,cmap='gray')
    # axx[2,1].axis('off')
    # axx[2,1].set_title('HSV white', fontsize=12)
    #____________________________________________________________________________
    #____________________________________________________________________________


    # res_image=np.zeros_like(thresh_binary_images2)
    # res_image[(thresh_binary_images2==1 & sobelMag_sobelAbs==0)]=1

    # axx[2,1].imshow(res_image,cmap='gray')
    # #print('added all shape:'+str(added_binary_images.shape)+' '+str(added_binary_images.dtype))
    # axx[2,1].axis('off')
    # axx[2,1].set_title('compare result', fontsize=15)
    #------------------------------------------------------
    ff, bxx = plt.subplots(4, 3, figsize=(20,10))
    ff.subplots_adjust(hspace = .1, wspace=0.01)

    bxx[0,0].imshow(exampleImg_sobelDir,cmap='gray')
    bxx[0,0].axis('off')
    bxx[0,0].set_title('Sobel Direction', fontsize=15)
    #-------------------------------------------------------------------
    bxx[0,1].imshow(exampleImg_sobelAbs,cmap='gray')
    bxx[0,1].axis('off')
    bxx[0,1].set_title('Sobel Absolute', fontsize=15)
    #--------------------------------------------------------------

    bxx[0,2].imshow(exampleImg_sobelMag,cmap='gray')
    bxx[0,2].axis('off')
    bxx[0,2].set_title('Sobel Magnitude', fontsize=15)


    bxx[1,0].imshow(sobelMag_sobelAbs,cmap='gray')
    bxx[1,0].axis('off')
    bxx[1,0].set_title('Sobel mag+abs', fontsize=12)
    #-------------------------------------------------------------------
    bxx[1,1].imshow(sobelAbs_sobelDir,cmap='gray')
    bxx[1,1].axis('off')
    bxx[1,1].set_title('Sobel abs+dir', fontsize=12)
    #-------------------------------------------------------------------



    bxx[1,2].imshow(sobelMag_sobelDir,cmap='gray')
    bxx[1,2].axis('off')
    bxx[1,2].set_title('Sobel Direction+Sobel magnitude', fontsize=12)

    bxx[2,0].imshow(final_binary,cmap='gray')
    bxx[2,0].axis('off')
    bxx[2,0].set_title('hls sobel', fontsize=12)

    bxx[2,1].imshow(kanali['edge_pos'],cmap='gray')
    bxx[2,1].axis('off')
    bxx[2,1].set_title('edge_pos', fontsize=12)

    #------------------------------------------------------------------
    # _, sxbinary = threshold(hls_hls[:, :, 1], thresh=(120, 255))
    bxx[2,2].imshow(kanali['edge_neg'],cmap='gray')
    bxx[2,2].axis('off')
    bxx[2,2].set_title('edge_neg', fontsize=12)

    lista=[exampleImg_sobelDir,exampleImg_sobelAbs,exampleImg_sobelMag,sobelMag_sobelAbs,sobelAbs_sobelDir,sobelMag_sobelDir,final_binary,kanali['edge_pos'],kanali['edge_neg']]
    rez_slika=IPF.make_binary_stack2(lista)

    bxx[3,0].imshow(rez_slika,cmap='gray')
    bxx[3,0].axis('off')
    bxx[3,0].set_title('added all together (MAX: '+str(max_value_sobel)+')', fontsize=12)


    threshold_value=2
    thresh_binary_image=IPF.threshold_binary_stack(rez_slika,threshold_value)
    bxx[3,1].imshow(thresh_binary_image,cmap='gray')
    bxx[3,1].axis('off')
    bxx[3,1].set_title('added all together THRESHOLD '+str(threshold_value), fontsize=12)




    sharpen_img=np.copy(exampleImg_unwarp)

    gausian_img=cv2.GaussianBlur(sharpen_img,(5,5),0)
    sharpened=cv2.addWeighted(exampleImg_unwarp,1.5,gausian_img,-0.5,0)
    bxx[3,2].imshow(sharpened,cmap='gray')
    bxx[3,2].axis('off')
    bxx[3,2].set_title('sharpened ', fontsize=12)
    # threshold_value2=3
    # thresh_binary_imagee=IPF.threshold_binary_stack(rez_slika,threshold_value2)
    # bxx[3,2].imshow(thresh_binary_imagee,cmap='gray')
    # bxx[3,2].axis('off')
    # bxx[3,2].set_title('added all together THRESHOLD '+str(threshold_value2), fontsize=12)



    ff, cxx = plt.subplots(4, 3, figsize=(20,10))
    ff.subplots_adjust(hspace = .1, wspace=0.01)

    cxx[0,0].imshow(rez_slika,cmap='gray')
    cxx[0,0].axis('off')
    cxx[0,0].set_title('SOBEL added all together', fontsize=12)

    max_value=np.max(rez_slika)
    threshold_value=IPF.threshold_equation(max_value)
    thresh_binary_image=IPF.threshold_binary_stack(rez_slika,threshold_value)

    cxx[0,1].imshow(thresh_binary_image,cmap='gray')
    cxx[0,1].axis('off')
    cxx[0,1].set_title('SOBEL with calc THRESHOLD: '+str(threshold_value), fontsize=12)


    cxx[1,0].imshow(stacked_binary_image_white,cmap='gray')
    cxx[1,0].axis('off')
    cxx[1,0].set_title('WHITE added all together', fontsize=12)

    # combined_white_sobel=np.zeros_like(thresh_binary_image)
    # combined_white_sobel+=thresh_binary_image
    # combined_white_sobel+=stacked_binary_image_white
    stacked_binary_white_sobel=thresh_binary_image+stacked_binary_image_white

    cxx[1,1].imshow(stacked_binary_white_sobel,cmap='gray')
    cxx[1,1].axis('off')
    cxx[1,1].set_title('WHITE + SOBEL added', fontsize=12)

    max_value2=np.max(stacked_binary_white_sobel)
    threshold_value2=IPF.threshold_equation(max_value2)
    stacked_thresholded_white_sobel=IPF.threshold_binary_stack(stacked_binary_white_sobel,threshold_value2)
    cxx[1,2].imshow(stacked_thresholded_white_sobel,cmap='gray')
    cxx[1,2].axis('off')
    cxx[1,2].set_title('WHITE + SOBEL calc thresholded: '+str(threshold_value2), fontsize=12)

    cxx[2,0].imshow(stacked_binary_image_yellow,cmap='gray')
    cxx[2,0].axis('off')
    cxx[2,0].set_title('YELLOW added all together', fontsize=12)


    stacked_binary_yellow_sobel=thresh_binary_image+stacked_binary_image_yellow
    cxx[2,1].imshow(stacked_binary_yellow_sobel,cmap='gray')
    cxx[2,1].axis('off')
    cxx[2,1].set_title('YELLOW + SOBEL added', fontsize=12)

    max_value3=np.max(stacked_binary_image_yellow)
    threshold_value3=IPF.threshold_equation(max_value3)
    stacked_thresholded_yellow_sobel=IPF.threshold_binary_stack(stacked_binary_image_yellow,threshold_value3)
    cxx[2,2].imshow(stacked_thresholded_yellow_sobel,cmap='gray')
    cxx[2,2].axis('off')
    cxx[2,2].set_title('YELLOW + SOBEL  calc thresholded: '+str(threshold_value3), fontsize=12)

    #------------------------------------------------------
    plt.show()

    return

###################################################################################################
if __name__ == "__main__":
    main()
