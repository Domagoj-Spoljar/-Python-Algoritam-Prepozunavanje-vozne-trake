import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import Image_processing_functions as IPF
import sys

if len(sys.argv) == 2:
    img_arg=sys.argv[1]
    print('Processing image: '+img_arg)
else:
    img_arg='frame100'
    print('Processing default image: '+img_arg)
dashcam_image_path = '/home/profesor/Documents/[ADAS]_Finding_Lanes/dashcam_driving/'

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
exampleImg_LBThresh = IPF.lab_threshold(exampleImg_unwarp,thresh=(0,255),color='b')
exampleImg_LLBThresh = IPF.lab_threshold(exampleImg_unwarp,thresh=(190,255),color='l')
exampleImg_RRGBThresh = IPF.rgb_thresh(exampleImg_unwarp,color=0)
exampleImg_GRGBThresh = IPF.rgb_thresh(exampleImg_unwarp,color=1)
exampleImg_BRGBThresh = IPF.rgb_thresh(exampleImg_unwarp,color=2)

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
#--------------------------------------------------------------

ffx, axy = plt.subplots(3, 4, figsize=(20,10))
ffx.subplots_adjust(hspace = .1, wspace=0.01)

#-------------------------------------------------------------
axy[1,1].imshow(exampleImg_SThresh,cmap='gray')
axy[1,1].axis('off')
axy[1,1].set_title('HLS S-channel thresholded', fontsize=15)

#------------------------------------------------------------------

axy[1,0].imshow(exampleImg_LThresh,cmap='gray')
axy[1,0].axis('off')
axy[1,0].set_title('HLS L-channel thresholded', fontsize=15)

#------------------------------------------------------------------

axy[2,1].imshow(exampleImg_LBThresh,cmap='gray')
axy[2,1].axis('off')
axy[2,1].set_title('LAB B-channel thresholded', fontsize=15)

#------------------------------------------------------------------
axy[2,0].imshow(exampleImg_LLBThresh,cmap='gray')
axy[2,0].axis('off')
axy[2,0].set_title('LAB L-channel thresholded', fontsize=15)

#------------------------------------------------------------------

axy[0,3].imshow(sobelMag_sobelDir,cmap='gray')
axy[0,3].axis('off')
axy[0,3].set_title('Sobel Direction+Sobel magnitude', fontsize=15)
#-------------------------------------------------------------------
axy[0,0].imshow(exampleImg_RRGBThresh,cmap='gray')
axy[0,0].axis('off')
axy[0,0].set_title('RGB r thresholded', fontsize=15)
#-------------------------------------------------------------------
axy[0,1].imshow(exampleImg_GRGBThresh,cmap='gray')
axy[0,1].axis('off')
axy[0,1].set_title('RGB g thresholded', fontsize=15)
#-------------------------------------------------------------------
axy[0,2].imshow(exampleImg_BRGBThresh,cmap='gray')
axy[0,2].axis('off')
axy[0,2].set_title('RGB b thresholded', fontsize=15)
#-------------------------------------------------------------------
axy[1,3].imshow(sobelMag_sobelAbs,cmap='gray')
axy[1,3].axis('off')
axy[1,3].set_title('Sobel mag+abs', fontsize=15)
#-------------------------------------------------------------------
axy[2,3].imshow(sobelAbs_sobelDir,cmap='gray')
axy[2,3].axis('off')
axy[2,3].set_title('Sobel abs+dir', fontsize=15)
#-------------------------------------------------------------------

#-------------------------------------------------------------------
combined_HLSl_LABb = np.zeros_like(exampleImg_LBThresh)
combined_HLSl_LABb[((exampleImg_LBThresh == 1) | (exampleImg_LThresh == 1))] = 1

combined_HLSs_LABb = np.zeros_like(exampleImg_SThresh)
combined_HLSs_LABb[((exampleImg_LBThresh == 1) | (exampleImg_SThresh == 1))] = 1

combined_HLSl_SOBELabs = np.zeros_like(exampleImg_LThresh)
combined_HLSl_SOBELabs[((exampleImg_LThresh==1)|(exampleImg_sobelAbs==1))]=1

combined_HLSl_SOBELmag = np.zeros_like(exampleImg_LThresh)
combined_HLSl_SOBELmag[((exampleImg_LThresh==1)|(exampleImg_sobelMag==1))]=1

combined_HLSl_HLSs = np.zeros_like(exampleImg_LThresh)
combined_HLSl_HLSs[((exampleImg_LThresh==1)|(exampleImg_SThresh==1))]=1



ff, axx = plt.subplots(3, 3, figsize=(20,10))
ff.subplots_adjust(hspace = .1, wspace=0.01)

axx[0,0].imshow(exampleImg_unwarp)
axx[0,0].axis('off')
axx[0,0].set_title('Unwarped Image', fontsize=15)

axx[0,1].imshow(combined_HLSl_LABb,cmap='gray')
axx[0,1].axis('off')
axx[0,1].set_title('LAB-B + HLS-L', fontsize=15)

axx[0,2].imshow(combined_HLSs_LABb,cmap='gray')
axx[0,2].axis('off')
axx[0,2].set_title('LAB-B + HLS-S', fontsize=15)

axx[1,0].imshow(combined_HLSl_SOBELabs,cmap='gray')
axx[1,0].axis('off')
axx[1,0].set_title('HLS-L + sobel abs', fontsize=15)

axx[1,1].imshow(combined_HLSl_SOBELmag,cmap='gray')
axx[1,1].axis('off')
axx[1,1].set_title('HLS-L + sobel mag', fontsize=15)

axx[1,2].imshow(combined_HLSl_HLSs,cmap='gray')
axx[1,2].axis('off')
axx[1,2].set_title('HLS-L + HLS-s', fontsize=15)
#------------------------------------------------------

#------------------------------------------------------
plt.show()
