import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
#import Lane_find_functions as ff
#from ipywidgets import interact, interactive, fixed
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#from camera_calibration import mtx, dist
#import perspective_transform.py as pt

mtx = np.array([[1.15694035e+03, 0.00000000e+00, 6.65948597e+02],[0.00000000e+00, 1.15213869e+03, 3.88785178e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float64)
dist = np.array([-2.37636612e-01, -8.54129776e-02, -7.90955950e-04, -1.15908700e-04, 1.05741395e-01],np.float64)

def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale === or LAB L channel
    gray = (cv2.cvtColor(img, cv2.COLOR_RGB2Lab))[:,:,0]
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    binary_output = sxbinary # Remove this line
    return binary_output

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude
    mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sxbinary)
    return binary_output

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_thresh(img, sobel_kernel=7, thresh=(0, 0.09)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def hls_sthresh(img, thresh=(125, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(hls[:,:,2])
    binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def hls_lthresh(img, thresh=(220, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def lab_bthresh(img, thresh=(190,255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def lab_lthresh(img, thresh=(190,255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab_b = lab[:,:,0]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def rgb_thresh(img, thresh=(200, 255), color = 0):
    # 1) Convert to HLS color space
    #rgb = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(img[:,:,color])
    binary_output[(img[:,:,color] > thresh[0]) & (img[:,:,color] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def sliding_window_polyfit(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

    #print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


#exampleImg = cv2.imread('./test_images/frame2.jpg')
exampleImg = cv2.imread('/home/profesor/Documents/[ADAS]_Finding_Lanes/dashcam_driving/frame105.jpg')
#exampleImg = cv2.imread('/home/profesor/Documents/test_files/roma/IRC04510/IMG00075.jpg')
exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)

exampleImg_undistort = undistort(exampleImg)


h,w = exampleImg_undistort.shape[:2]

# define source and destination points for transform

# original values
src = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])
# src = np.float32([(550,430),
#                   (730,430),
#                   (200,622),
#                   (1080,622)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

exampleImg_unwarp, M, Minv = unwarp(exampleImg_undistort, src, dst)

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
exampleImg_sobelAbs = abs_sobel_thresh(exampleImg_unwarp, 'x', min_thresh, max_thresh)

min_thresh2=25
max_thresh2=255
kernel_size=25
exampleImg_sobelMag = mag_thresh(exampleImg_unwarp, kernel_size, (min_thresh2, max_thresh2))

min_thresh3=0
max_thresh3=0.11
kernel_size2=7
exampleImg_sobelDir = dir_thresh(exampleImg_unwarp, kernel_size2, (min_thresh3, max_thresh3))

sobelMag_sobelDir = np.zeros_like(exampleImg_sobelMag)
sobelMag_sobelDir[((exampleImg_sobelMag == 1) & (exampleImg_sobelDir == 1))] = 1

sobelAbs_sobelDir = np.zeros_like(exampleImg_sobelAbs)
sobelAbs_sobelDir[((exampleImg_sobelAbs == 1) & (exampleImg_sobelDir == 1))] = 1

sobelMag_sobelAbs = np.zeros_like(exampleImg_sobelMag)
sobelMag_sobelAbs[((exampleImg_sobelMag == 1) & (exampleImg_sobelAbs == 1))] = 1

exampleImg_SThresh = hls_sthresh(exampleImg_unwarp)
exampleImg_LThresh = hls_lthresh(exampleImg_unwarp)
exampleImg_LBThresh = lab_bthresh(exampleImg_unwarp)
exampleImg_LLBThresh = lab_lthresh(exampleImg_unwarp)
exampleImg_RRGBThresh = rgb_thresh(exampleImg_unwarp,color=0)
exampleImg_GRGBThresh = rgb_thresh(exampleImg_unwarp,color=1)
exampleImg_BRGBThresh = rgb_thresh(exampleImg_unwarp,color=2)

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
