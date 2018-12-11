import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os

mtx = np.array([[1.15694035e+03, 0.00000000e+00, 6.65948597e+02],[0.00000000e+00, 1.15213869e+03, 3.88785178e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float64)
dist = np.array([-2.37636612e-01, -8.54129776e-02, -7.90955950e-04, -1.15908700e-04, 1.05741395e-01],np.float64)


#Function that undistorts image with calculated mtx and dist
def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

#Function that converts Region of Interest (ROI) defined with "src" points
#and makes perspective transfrom to apply "birds eye view" on process_image
#also returns inverse matrix (Minv) for later usage with ploting lines to original image
def unwarp(img, src, dst):
    h,w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

#Function that converts plot into image
def plot_fit_onto_img(img, fit, plot_color):
    if fit is None:
        return img
    new_img = np.copy(img)
    h = new_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(new_img, np.int32([pts]), isClosed=False, color=plot_color, thickness=8)
    return new_img

# Function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def sobel_abs_thresh(img, orient='x', thresh_min=25, thresh_max=255):
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

# Function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def sobel_mag_thresh(img, sobel_kernel=25, mag_thresh=(25, 255)):

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

# Function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def sobel_dir_thresh(img, sobel_kernel=7, thresh=(0, 0.09)):
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

#Function that thresholds one RGB channel and returns binary image
def rgb_thresh(img, thresh=(200, 255), color = 0):
    # 1) Convert to HLS color space
    #rgb = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(img[:,:,color])
    binary_output[(img[:,:,color] > thresh[0]) & (img[:,:,color] <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

#Function that thresholds one HLS channel and returns binary image
def hls_threshold(img, thresh=(125, 255), color='h'):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    if color == 's':
        binary_output = np.zeros_like(hls[:,:,2])
        binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    elif color == 'l':
        hls_l = hls[:,:,1]
        hls_l = hls_l*(255/np.max(hls_l))
        binary_output = np.zeros_like(hls_l)
        binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    elif color == 'h':
        
    else:
        return None
    # 3) Return a binary image of threshold result
    return binary_output
