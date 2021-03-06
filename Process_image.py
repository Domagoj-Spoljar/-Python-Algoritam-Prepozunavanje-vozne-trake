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
    #dashcam_image_path = './Test_images/dashcam_driving/'
    #dashcam_image_path = './Test_images/challnege_video/'
    #dashcam_image_path = './Test_images/harder_challenge_video/'
    #dashcam_image_path = './Test_images/project_video/'

    #exampleImg = cv2.imread('./test_images/frame2.jpg')
    #exampleImg = cv2.imread('/home/profesor/Documents/[ADAS]_Finding_Lanes/dashcam_driving/frame501.jpg')
    exampleImg = cv2.imread(dashcam_image_path+img_arg+".jpg")
    #exampleImg = cv2.imread('/home/profesor/Documents/test_files/roma/IRC04510/IMG00075.jpg')
    exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)

    exampleImg_undistort = IPF.undistort(exampleImg)


    h,w = exampleImg_undistort.shape[:2]
    # define source and destination points for transform
    src,dst = IPF.unwarp_points(h,w)

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
    exampleImg_LBThresh = IPF.lab_threshold(exampleImg_unwarp,thresh=(150,255),color='b')
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


    #_______________________________________________________________________________________________________________________________
    #_______________________________________________________________________________________________________________________________
    # Visualize undistortion


    f, ax = plt.subplots(4, 4, figsize=(20,10))
    f.subplots_adjust(hspace = .1, wspace=0.01)

    ax[0,0].imshow(exampleImg)
    ax[0,0].axis('off')
    ax[0,0].set_title('Original Image', fontsize=15)

    ax[0,1].imshow(exampleImg_undistort)
    ax[0,1].axis('off')
    ax[0,1].set_title('Undistorted Image', fontsize=15)

    ax[0,2].imshow(exampleImg_undistort)
    x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
    y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
    ax[0,2].plot(x, y, color='#ff0000', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
    ax[0,2].axis('off')
    ax[0,2].set_title('Undistorted Image with ROI', fontsize=15)

    ax[0,3].imshow(exampleImg_unwarp)
    ax[0,3].axis('off')
    ax[0,3].set_title('Unwarped Image', fontsize=15)

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

    ax[1,3].imshow(exampleImg_sobelAbs,cmap='gray')
    ax[1,3].axis('off')
    ax[1,3].set_title('Sobel Absolute', fontsize=15)
    #--------------------------------------------------------------

    ax[2,3].imshow(exampleImg_sobelMag,cmap='gray')
    ax[2,3].axis('off')
    ax[2,3].set_title('Sobel Magnitude', fontsize=15)
    #--------------------------------------------------------------

    ax[3,3].imshow(exampleImg_sobelDir,cmap='gray')
    ax[3,3].axis('off')
    ax[3,3].set_title('Sobel Direction', fontsize=15)


    #_______________________________________________________________________________________________________________________________
    #_______________________________________________________________________________________________________________________________
    #--------------------------------------------------------------

    ffx, axy = plt.subplots(4, 4, figsize=(20,10))
    ffx.subplots_adjust(hspace = .1, wspace=0.01)

    #-------------------------------------------------------------
    axy[1,1].imshow(exampleImg_SThresh,cmap='gray')
    axy[1,1].axis('off')
    axy[1,1].set_title('HLS S-channel thresholded', fontsize=12)

    #------------------------------------------------------------------

    axy[1,0].imshow(exampleImg_LThresh,cmap='gray')
    axy[1,0].axis('off')
    axy[1,0].set_title('HLS L-channel thresholded', fontsize=12)

    #------------------------------------------------------------------

    axy[2,1].imshow(exampleImg_LBThresh,cmap='gray')
    axy[2,1].axis('off')
    axy[2,1].set_title('LAB B-channel thresholded', fontsize=12)

    #------------------------------------------------------------------
    axy[2,0].imshow(exampleImg_LLBThresh,cmap='gray')
    axy[2,0].axis('off')
    axy[2,0].set_title('LAB L-channel thresholded', fontsize=12)

    #------------------------------------------------------------------

    axy[0,3].imshow(sobelMag_sobelDir,cmap='gray')
    axy[0,3].axis('off')
    axy[0,3].set_title('Sobel Direction+Sobel magnitude', fontsize=12)
    #-------------------------------------------------------------------
    axy[0,0].imshow(exampleImg_RRGBThresh,cmap='gray')
    axy[0,0].axis('off')
    axy[0,0].set_title('RGB r thresholded', fontsize=12)
    #-------------------------------------------------------------------
    axy[0,1].imshow(exampleImg_GRGBThresh,cmap='gray')
    axy[0,1].axis('off')
    axy[0,1].set_title('RGB g thresholded', fontsize=12)
    #-------------------------------------------------------------------
    axy[0,2].imshow(exampleImg_BRGBThresh,cmap='gray')
    axy[0,2].axis('off')
    axy[0,2].set_title('RGB b thresholded', fontsize=12)
    #-------------------------------------------------------------------
    axy[1,2].imshow(sobelMag_sobelAbs,cmap='gray')
    axy[1,2].axis('off')
    axy[1,2].set_title('Sobel mag+abs', fontsize=12)
    #-------------------------------------------------------------------
    axy[1,3].imshow(sobelAbs_sobelDir,cmap='gray')
    axy[1,3].axis('off')
    axy[1,3].set_title('Sobel abs+dir', fontsize=12)
    #-------------------------------------------------------------------
    axy[2,2].imshow(added_binary_images,cmap='gray')
    axy[2,2].axis('off')
    axy[2,2].set_title('added all binary images', fontsize=12)
    #-------------------------------------------------------------------
    axy[2,3].imshow(thresh_binary_images2,cmap='gray')
    axy[2,3].axis('off')
    axy[2,3].set_title('thresholded all binary images 2', fontsize=12)
    #-------------------------------------------------------------------
    axy[3,0].imshow(thresh_binary_images3,cmap='gray')
    axy[3,0].axis('off')
    axy[3,0].set_title('thresholded all binary images 3', fontsize=12)
    #-------------------------------------------------------------------
    axy[3,1].imshow(thresh_binary_images4,cmap='gray')
    axy[3,1].axis('off')
    axy[3,1].set_title('thresholded all binary images 4', fontsize=12)
    #-------------------------------------------------------------------
    axy[3,2].imshow(thresh_binary_images5,cmap='gray')
    axy[3,2].axis('off')
    axy[3,2].set_title('thresholded all binary images 5', fontsize=12)
    #-------------------------------------------------------------------
    axy[3,3].imshow(thresh_binary_images6,cmap='gray')
    axy[3,3].axis('off')
    axy[3,3].set_title('thresholded all binary images 6', fontsize=12)
    #-------------------------------------------------------------------
    # ctr0=0
    # ctr1=0
    # ctr2=0
    # ctr3=0
    # ctr4=0
    # ctr5=0
    # ctr6=0
    # ctr7=0
    # ctr8=0
    # ctr9=0
    # ctr10=0
    #
    # for x in range(thresh_binary_images2.shape[0]):
    #     for y in range(thresh_binary_images2.shape[1]):
    #         if thresh_binary_images2[x,y]==1:
    #             ctr0+=1
    #         if thresh_binary_images2[x,y] == exampleImg_RRGBThresh[x,y]:
    #             ctr1+=1
    #         else: ctr1-=1
    #         if thresh_binary_images2[x,y]==exampleImg_GRGBThresh[x,y]:
    #             ctr2+=1
    #         else: ctr2-=1
    #         if thresh_binary_images2[x,y]==exampleImg_BRGBThresh[x,y]:
    #             ctr3+=1
    #         else: ctr3-=1
    #         if thresh_binary_images2[x,y]==sobelMag_sobelDir[x,y]:
    #             ctr4+=1
    #         else: ctr4-=1
    #         if thresh_binary_images2[x,y]==exampleImg_LThresh[x,y]:
    #             ctr5+=1
    #         else: ctr5-=1
    #         if thresh_binary_images2[x,y]==exampleImg_SThresh[x,y]:
    #             ctr6+=1
    #         else: ctr6-=1
    #         if thresh_binary_images2[x,y]==sobelMag_sobelAbs[x,y]:
    #             ctr7+=1
    #         else: ctr7-=1
    #         if thresh_binary_images2[x,y]==sobelAbs_sobelDir[x,y]:
    #             ctr8+=1
    #         else: ctr8-=1
    #         if thresh_binary_images2[x,y]==exampleImg_LBThresh[x,y]:
    #             ctr9+=1
    #         else: ctr9-=1
    #         if thresh_binary_images2[x,y]==exampleImg_LLBThresh[x,y]:
    #             ctr10+=1
    #         else: ctr10-=1
    #
    # print('REF image total 1 count='+str(ctr0))
    # print('RGB R= '+str(ctr1))
    # print('RGB G= '+str(ctr2))
    # print('RGB B= '+str(ctr3))
    # print('Mag+Dir= '+str(ctr4))
    # print('HLS L= '+str(ctr5))
    # print('HLS S= '+str(ctr6))
    # print('Mag+Abs= '+str(ctr7))
    # print('Abs+Dir= '+str(ctr8))
    # print('LAB B= '+str(ctr9))
    # print('LAB L= '+str(ctr10))
    #
    # razlika=[(ctr1,'RGB-R'), (ctr2,'RGB-G'),(ctr3,'RGB-B'),(ctr4,'Mag+Dir'),(ctr5,'HLS-L'),(ctr6,'HLS-S'),(ctr7,'Mag+Abs'),(ctr8,'Abs+Dir'),(ctr9,'LAB-B'),(ctr10,'LAB-L')]
    # razlika.sort(key=lambda x: x[0], reverse=True)
    # print('Best combinations: '+razlika[0][1]+' OR '+razlika[1][1])

    #**********************************************************************************************************************************************************************
    #**********************************************************************************************************************************************************************

    # ctr0=0
    # ctr1=0
    # ctr2=0
    # ctr3=0
    # ctr4=0
    # ctr5=0
    # ctr6=0
    # ctr7=0
    # ctr8=0
    # ctr9=0
    # ctr10=0
    #
    # for x in range(thresh_binary_images2.shape[0]):
    #     for y in range(thresh_binary_images2.shape[1]):
    #         if thresh_binary_images2[x,y]==1:
    #             ctr0+=1
    #
    #         if thresh_binary_images2[x,y]==1 & exampleImg_RRGBThresh[x,y]==1:
    #             ctr1+=1
    #         elif thresh_binary_images2[x,y]==0 & exampleImg_RRGBThresh[x,y]==1:
    #             ctr1-=1
    #         if thresh_binary_images2[x,y]==1 & exampleImg_GRGBThresh[x,y]==1:
    #             ctr2+=1
    #         elif thresh_binary_images2[x,y]==0 & exampleImg_GRGBThresh[x,y]==1:
    #             ctr2-=1
    #         if thresh_binary_images2[x,y]==1 & exampleImg_BRGBThresh[x,y]==1:
    #             ctr3+=1
    #         elif thresh_binary_images2[x,y]==0 & exampleImg_BRGBThresh[x,y]==1:
    #             ctr3-=1
    #         if thresh_binary_images2[x,y]==1 & sobelMag_sobelDir[x,y]==1:
    #             ctr4+=1
    #         elif thresh_binary_images2[x,y]==0 & sobelMag_sobelDir[x,y]==1:
    #             ctr4-=1
    #         if thresh_binary_images2[x,y]==1 & exampleImg_LThresh[x,y]==1:
    #             ctr5+=1
    #         elif thresh_binary_images2[x,y]==0 & exampleImg_LThresh[x,y]==1:
    #             ctr5-=1
    #         if thresh_binary_images2[x,y]==1 & exampleImg_SThresh[x,y]==1:
    #             ctr6+=1
    #         elif thresh_binary_images2[x,y]==0 & exampleImg_SThresh[x,y]==1:
    #             ctr6-=1
    #         if thresh_binary_images2[x,y]==1 & sobelMag_sobelAbs[x,y]==1:
    #             ctr7+=1
    #         elif thresh_binary_images2[x,y]==0 & sobelMag_sobelAbs[x,y]==1:
    #             ctr7-=1
    #         if thresh_binary_images2[x,y]==1 & sobelAbs_sobelDir[x,y]==1:
    #             ctr8+=1
    #         elif thresh_binary_images2[x,y]==0 & sobelAbs_sobelDir[x,y]==1:
    #             ctr8-=1
    #         if thresh_binary_images2[x,y]==1 & exampleImg_LBThresh[x,y]==1:
    #             ctr9+=1
    #         elif thresh_binary_images2[x,y]==0 & exampleImg_LBThresh[x,y]==1:
    #             ctr9-=1
    #         if thresh_binary_images2[x,y]==1 & exampleImg_LLBThresh[x,y]==1:
    #             ctr10+=1
    #         elif thresh_binary_images2[x,y]==0 & exampleImg_LLBThresh[x,y]==1:
    #             ctr10-=1
    #
    # print('REF image total 1 count='+str(ctr0))
    # print('RGB R= '+str(ctr1))
    # print('RGB G= '+str(ctr2))
    # print('RGB B= '+str(ctr3))
    # print('Mag+Dir= '+str(ctr4))
    # print('HLS L= '+str(ctr5))
    # print('HLS S= '+str(ctr6))
    # print('Mag+Abs= '+str(ctr7))
    # print('Abs+Dir= '+str(ctr8))
    # print('LAB B= '+str(ctr9))
    # print('LAB L= '+str(ctr10))
    #
    # razlika=[(ctr0-ctr1,'RGB-R'), (ctr0-ctr2,'RGB-G'),(ctr0-ctr3,'RGB-B'),(ctr0-ctr4,'Mag+Dir'),(ctr0-ctr5,'HLS-L'),(ctr0-ctr6,'HLS-S'),(ctr0-ctr7,'Mag+Abs'),(ctr0-ctr8,'Abs+Dir'),(ctr0-ctr9,'LAB-B'),(ctr0-ctr10,'LAB-L')]
    # razlika.sort(key=lambda x: x[0])
    # print('Best combinations: '+razlika[0][1]+' OR '+razlika[1][1])



    #**********************************************************************************************************************************************************************
    #**********************************************************************************************************************************************************************
    #____________________________________________________________________________
    #____________________________________________________________________________

    # combined_HLSl_LABb = np.zeros_like(exampleImg_LBThresh)
    # combined_HLSl_LABb[((exampleImg_LBThresh == 1) | (exampleImg_LThresh == 1))] = 1
    #
    # combined_HLSs_LABb = np.zeros_like(exampleImg_SThresh)
    # combined_HLSs_LABb[((exampleImg_LBThresh == 1) | (exampleImg_SThresh == 1))] = 1
    #
    # combined_HLSl_SOBELabs = np.zeros_like(exampleImg_LThresh)
    # combined_HLSl_SOBELabs[((exampleImg_LThresh==1)|(exampleImg_sobelAbs==1))]=1
    #
    # combined_HLSl_SOBELmag = np.zeros_like(exampleImg_LThresh)
    # combined_HLSl_SOBELmag[((exampleImg_LThresh==1)|(exampleImg_sobelMag==1))]=1
    #
    # combined_HLSl_HLSs = np.zeros_like(exampleImg_LThresh)
    # combined_HLSl_HLSs[((exampleImg_LThresh==1)|(exampleImg_SThresh==1))]=1
    #
    #
    #
    # ff, axx = plt.subplots(3, 3, figsize=(20,10))
    # ff.subplots_adjust(hspace = .1, wspace=0.01)
    #
    # axx[0,0].imshow(exampleImg_unwarp)
    # axx[0,0].axis('off')
    # axx[0,0].set_title('Unwarped Image', fontsize=15)
    #
    # axx[0,1].imshow(combined_HLSl_LABb,cmap='gray')
    # axx[0,1].axis('off')
    # axx[0,1].set_title('LAB-B + HLS-L', fontsize=15)
    #
    # axx[0,2].imshow(combined_HLSs_LABb,cmap='gray')
    # axx[0,2].axis('off')
    # axx[0,2].set_title('LAB-B + HLS-S', fontsize=15)
    #
    # axx[1,0].imshow(combined_HLSl_SOBELabs,cmap='gray')
    # axx[1,0].axis('off')
    # axx[1,0].set_title('HLS-L + sobel abs', fontsize=15)
    #
    # axx[1,1].imshow(combined_HLSl_SOBELmag,cmap='gray')
    # axx[1,1].axis('off')
    # axx[1,1].set_title('HLS-L + sobel mag', fontsize=15)
    #
    # axx[1,2].imshow(combined_HLSl_HLSs,cmap='gray')
    # axx[1,2].axis('off')
    # axx[1,2].set_title('HLS-L + HLS-s', fontsize=15)
    # #------------------------------------------------------
    # axx[2,0].imshow(added_binary_images,cmap='gray')
    # #print('added all shape:'+str(added_binary_images.shape)+' '+str(added_binary_images.dtype))
    # axx[2,0].axis('off')
    # axx[2,0].set_title('added all binary images', fontsize=15)
    #----------------------------------------------------------
    #____________________________________________________________________________
    #____________________________________________________________________________


    # res_image=np.zeros_like(thresh_binary_images2)
    # res_image[(thresh_binary_images2==1 & sobelMag_sobelAbs==0)]=1

    # axx[2,1].imshow(res_image,cmap='gray')
    # #print('added all shape:'+str(added_binary_images.shape)+' '+str(added_binary_images.dtype))
    # axx[2,1].axis('off')
    # axx[2,1].set_title('compare result', fontsize=15)
    #------------------------------------------------------

    #------------------------------------------------------
    plt.show()

    return

###################################################################################################
if __name__ == "__main__":
    main()
