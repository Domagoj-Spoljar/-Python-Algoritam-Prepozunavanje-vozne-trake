import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import Image_processing_functions as IPF
import sys
import function_parameters as FP


def split_channels(image) :
        """
        returns a total of 7 channels :
        4 edge channels : all color edges (including the signs), yellow edges (including the signs)
        3 color channels : yellow and white (2 different thresholds are used for white)
        """
        binary = {}

        # thresholding parameters for various color channels and Sobel x-gradients
        h_thresh=(15, 35)
        #s_thresh=(75, 255)
        s_thresh=(30, 255)
        v_thresh=(175,255)
        vx_thresh = (20, 120)
        sx_thresh=(10, 100)

        img = np.copy(image)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]

        # Sobel x for v-channel
        sobelx_pos = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
        sobelx_neg = np.copy(sobelx_pos)
        sobelx_pos[sobelx_pos<=0] = 0
        sobelx_neg[sobelx_neg>0] = 0
        sobelx_neg = np.absolute(sobelx_neg)
        scaled_sobel_pos = np.uint8(255*sobelx_pos/np.max(sobelx_pos))
        scaled_sobel_neg = np.uint8(255*sobelx_neg/np.max(sobelx_neg))
        vxbinary_pos = np.zeros_like(v_channel)
        vxbinary_pos[(scaled_sobel_pos >= vx_thresh[0]) & (scaled_sobel_pos <= vx_thresh[1])] = 1
        binary['edge_pos'] = vxbinary_pos
        vxbinary_neg = np.zeros_like(v_channel)
        vxbinary_neg[(scaled_sobel_neg >= vx_thresh[0]) & (scaled_sobel_neg <= vx_thresh[1])] = 1
        binary['edge_neg'] = vxbinary_neg

        # Sobel x for s-channel
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
        sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))
        sxbinary_pos = np.zeros_like(s_channel)
        sxbinary_neg = np.zeros_like(s_channel)
        sxbinary_pos[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])
                     & (scaled_sobel_pos >= vx_thresh[0]-10) & (scaled_sobel_pos <= vx_thresh[1])]=1
        sxbinary_neg[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])
                     & (scaled_sobel_neg >= vx_thresh[0]-10) & (scaled_sobel_neg <= vx_thresh[1])]=1
        binary['yellow_edge_pos'] = sxbinary_pos
        binary['yellow_edge_neg'] = sxbinary_neg

        # color thresholds for selecting white lines
        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel >= v_thresh[0]+s_channel+20) & (v_channel <= v_thresh[1])] = 1
        binary['white_tight'] = np.copy(v_binary)
        v_binary[v_channel >= v_thresh[0]+s_channel] = 1
        binary['white_loose'] = v_binary

        # color threshold for selecting yellow lines
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1]) & (s_channel >= s_thresh[0])] = 1
        binary['yellow'] = h_binary

        return binary


def threshold(channel, thresh=(128,255), thresh_type=cv2.THRESH_BINARY):
#     # threshold the supplied channel
#     # channel : 2D array of the channel data
#     # thresh : 2D tupple of min and max threshold values
#     # thresh_type : what type of threshold to apply
#     # return : 2D thresholded data
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)

def binary_array(array, thresh, value=1):
    # turns an array into a binary array when between a threshold
    # array : numpy array to be converted to binary
    # thresh : threshold values between which a change in binary is stored.
    #          Threshold is inclusive
    # value : output value when between the supplied threshold
    # return : Binary array version of the supplied array

    # Is activation (1) between the threshold values (band-pass) or is it
    # outside the threshold values (band-stop)
    if value == 0:
        # Create a binary array the same size of as the input array
        # band-stop binary array
        binary = np.ones_like(array)
    else:
        # band-pass binary array
        binary = np.zeros_like(array)
        value = 1

    binary[(array >= thresh[0]) & (array <= thresh[1])] = value
    return binary

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):

    # get the edges in the horizontal direction
    sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
    # get the edges in the vertical direction
    sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))

    # Calculate the edge magnitudes
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return binary_array(mag, thresh)

def sobel(img_channel, orient='x', sobel_kernel=3):

    if orient == 'x':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)

    return sobel

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
exampleImg_LBThresh = IPF.lab_threshold(img_unwarp_inverted,thresh=(150,255),color='b')
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
thresh_binary_image,_=IPF.make_binary_stack(exampleImg_unwarp)
#thresh_binary_images3=IPF.binary_threshold(added_binary_images,3)
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

ffx, axy = plt.subplots(4, 4, figsize=(20,10))
ffx.subplots_adjust(hspace = .1, wspace=0.01)

#-------------------------------------------------------------
kanali = split_channels(exampleImg_unwarp)
axy[0,0].imshow(kanali['edge_pos'],cmap='gray')
axy[0,0].axis('off')
axy[0,0].set_title('edge_pos', fontsize=12)

#------------------------------------------------------------------
# _, sxbinary = threshold(hls_hls[:, :, 1], thresh=(120, 255))
axy[0,1].imshow(kanali['edge_neg'],cmap='gray')
axy[0,1].axis('off')
axy[0,1].set_title('edge_neg', fontsize=12)

# #------------------------------------------------------------------
# sxbinary = blur_gaussian(sxbinary, ksize=3)
axy[0,2].imshow(kanali['yellow_edge_pos'],cmap='gray')
axy[0,2].axis('off')
axy[0,2].set_title('yellow_edge_pos', fontsize=12)
# #------------------------------------------------------------------
axy[0,3].imshow(kanali['yellow_edge_neg'],cmap='gray')
axy[0,3].axis('off')
axy[0,3].set_title('yellow_edge_neg', fontsize=12)

# #------------------------------------------------------------------
axy[1,0].imshow(kanali['white_tight'],cmap='gray')
axy[1,0].axis('off')
axy[1,0].set_title('white_tight', fontsize=12)
# # #-------------------------------------------------------------------
axy[1,1].imshow(kanali['white_loose'],cmap='gray')
axy[1,1].axis('off')
axy[1,1].set_title('white_loose', fontsize=12)
# # #-------------------------------------------------------------------
axy[1,2].imshow(kanali['yellow'],cmap='gray')
axy[1,2].axis('off')
axy[1,2].set_title('yellow', fontsize=12)
# # #-------------------------------------------------------------------
imgA = np.dstack((kanali['edge_pos'], kanali['edge_neg'], kanali['white_tight']))
axy[1,3].imshow(imgA)
axy[1,3].axis('off')
axy[1,3].set_title('edge_pos=R edge_neg=G white_tight=B', fontsize=12)
# #-------------------------------------------------------------------
imgB = np.dstack((kanali['yellow_edge_pos'], kanali['yellow_edge_neg'], kanali['yellow']))
axy[2,0].imshow(imgB,cmap='gray')
axy[2,0].axis('off')
axy[2,0].set_title('yellow_edge_pos=B yellow_edge_neg=G yellow=R', fontsize=12)
# #-------------------------------------------------------------------
res_binary_edge=np.zeros_like(kanali['edge_pos'])

res_binary_edge = cv2.bitwise_or(kanali['edge_pos'], kanali['edge_neg'])
axy[2,1].imshow(res_binary_edge,cmap='gray')
axy[2,1].axis('off')
axy[2,1].set_title('edge_pos OR edge_neg', fontsize=12)
# #-------------------------------------------------------------------
res_binary_yellow=np.zeros_like(kanali['edge_pos'])
res_binary_yellow = cv2.bitwise_or(kanali['yellow_edge_neg'], kanali['yellow_edge_pos'])
axy[2,2].imshow(res_binary_yellow,cmap='gray')
axy[2,2].axis('off')
axy[2,2].set_title('yellow_edge_neg OR yellow_edge_pos', fontsize=12)
# #-------------------------------------------------------------------
res_binary_white=np.zeros_like(kanali['edge_pos'])
res_binary_white = cv2.bitwise_or(kanali['white_tight'], kanali['white_loose'])
axy[2,3].imshow(res_binary_white,cmap='gray')
axy[2,3].axis('off')
axy[2,3].set_title('white_loose OR white_tight', fontsize=12)
# #-------------------------------------------------------------------
res_binary_edgeANDwhite =np.zeros_like(kanali['edge_pos'])
res_binary_edgeANDwhite = cv2.bitwise_and(res_binary_edge, res_binary_white)
axy[3,0].imshow(res_binary_edgeANDwhite,cmap='gray')
axy[3,0].axis('off')
axy[3,0].set_title('res_binary_edge AND res_binary_white', fontsize=12)
# #-------------------------------------------------------------------
res_binary_edgeANDyellowedge =np.zeros_like(kanali['edge_pos'])
res_binary_edgeANDyellowedge = cv2.bitwise_and(res_binary_edge, res_binary_yellow)
axy[3,1].imshow(res_binary_edgeANDyellowedge,cmap='gray')
axy[3,1].axis('off')
axy[3,1].set_title('res_binary_edge AND res_binary_yellow_edge', fontsize=12)
# #-------------------------------------------------------------------
res_binary_edgeANDyellow =np.zeros_like(kanali['edge_pos'])
res_binary_edgeANDyellow = cv2.bitwise_and(res_binary_edge, kanali['yellow'])
axy[3,2].imshow(res_binary_edgeANDyellow,cmap='gray')
axy[3,2].axis('off')
axy[3,2].set_title('res_binary_edge AND res_binary_yellow', fontsize=12)
# #-------------------------------------------------------------------
res_binary_whiteORyellow=np.zeros_like(kanali['edge_pos'])
res_binary_whiteORyellow= cv2.bitwise_or(res_binary_edgeANDwhite, res_binary_edgeANDyellow)
axy[3,3].imshow(res_binary_whiteORyellow,cmap='gray')
axy[3,3].axis('off')
axy[3,3].set_title('white final + yellow final', fontsize=12)
# #-------------------------------------------------------------------
#
# #-------------------------------------------------------------------
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
ff, axx = plt.subplots(3, 4, figsize=(20,10))
ff.subplots_adjust(hspace = .1, wspace=0.01)

exampleImg_unwarp[:, :, 0] = cv2.equalizeHist(exampleImg_unwarp[:, :, 0])
exampleImg_unwarp[:, :, 1] = cv2.equalizeHist(exampleImg_unwarp[:, :, 1])
exampleImg_unwarp[:, :, 2] = cv2.equalizeHist(exampleImg_unwarp[:, :, 2])


axx[0,0].imshow(thresh_binary_images2,cmap='gray')
axx[0,0].axis('off')
axx[0,0].set_title('thresholded all binary images 2', fontsize=15)

axx[0,1].imshow(thresh_binary_images3,cmap='gray')
axx[0,1].axis('off')
axx[0,1].set_title('thresholded all binary images 3', fontsize=15)

axx[0,2].imshow(thresh_binary_images4,cmap='gray')
axx[0,2].axis('off')
axx[0,2].set_title('thresholded all binary images 4', fontsize=15)

axx[0,3].imshow(thresh_binary_images5,cmap='gray')
axx[0,3].axis('off')
axx[0,3].set_title('thresholded all binary images 5', fontsize=15)

added_binary_images=np.zeros_like(exampleImg_unwarp[:,:,0])
added_binary_images=kanali['edge_pos']+kanali['edge_neg']+kanali['yellow_edge_neg']+kanali['yellow_edge_pos']+kanali['white_tight']+kanali['white_loose']+kanali['yellow']
thresh_binary_image22=np.zeros_like(added_binary_images[:,:])
thresh_binary_image33=np.zeros_like(added_binary_images[:,:])
thresh_binary_image44=np.zeros_like(added_binary_images[:,:])
thresh_binary_image55=np.zeros_like(added_binary_images[:,:])
thresh_binary_image22[(added_binary_images[:,:] >= 2)] = 1
thresh_binary_image33[(added_binary_images[:,:] >= 3)] = 1
thresh_binary_image44[(added_binary_images[:,:] >= 4)] = 1
thresh_binary_image55[(added_binary_images[:,:] >= 5)] = 1


axx[1,0].imshow(thresh_binary_image22,cmap='gray')
axx[1,0].axis('off')
axx[1,0].set_title('thresholded all binary images 22', fontsize=15)

axx[1,1].imshow(thresh_binary_image33,cmap='gray')
axx[1,1].axis('off')
axx[1,1].set_title('thresholded all binary images 33', fontsize=15)

axx[1,2].imshow(thresh_binary_image44,cmap='gray')
axx[1,2].axis('off')
axx[1,2].set_title('thresholded all binary images 44', fontsize=15)

axx[1,3].imshow(thresh_binary_image55,cmap='gray')
axx[1,3].axis('off')
axx[1,3].set_title('thresholded all binary images 55', fontsize=15)

# lista=['rgb_r','hls_s','hls_l','lab_l','lab_b','sobel_mag','sobel_abs','sobel_dir']
lista=['hsv_white','hsv_yellow','white_tight','white_loose','yellow_edge_pos','yellow_edge_neg','yellow','edge_pos','hls_sobel']
stacked_binary_image,all_binary_images=IPF.make_binary_stack_custom(exampleImg_unwarp,lista)
axx[2,0].imshow(stacked_binary_image,cmap='gray')
axx[2,0].axis('off')
axx[2,0].set_title('test', fontsize=15)
print(len(all_binary_images))

edges = cv2.Canny(exampleImg_unwarp_R,120,170)
axx[2,1].imshow(edges,cmap='gray')
axx[2,1].axis('off')
axx[2,1].set_title('canny', fontsize=15)
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
#------------------------------------------------------

#------------------------------------------------------
plt.show()
