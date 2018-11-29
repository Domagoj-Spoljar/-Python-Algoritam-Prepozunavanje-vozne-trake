import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
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




def main():
    dashcam_image_path = '/home/profesor/Documents/[ADAS]_Finding_Lanes/dashcam_driving/'
    img_arg="frame"
    count = 100
    k=0
    #cv2.namedWindow('prikaz', cv2.WINDOW_NORMAL)
    while k is not 27:

        imgOriginal = cv2.imread(dashcam_image_path+img_arg+str(count)+".jpg")               # open image
        #cv2.imshow(img_arg+str(count)+".jpg", imgOriginal)
        cv2.namedWindow(img_arg+str(count)+".jpg", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(img_arg+str(count)+".jpg",1280,720)
        if imgOriginal is None:                             # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return

        exampleImg=np.copy(imgOriginal)
        exampleImg_undistort = undistort(exampleImg)
        #exampleImg_undistort = exampleImg
        h,w = exampleImg_undistort.shape[:2]
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

        #-----------------------------------------------------------
        min_thresh=50
        max_thresh=150
        exampleImg_sobelAbs = abs_sobel_thresh(exampleImg_unwarp, 'x', min_thresh, max_thresh)

        #--------------------------------------------------------------
        min_thresh2=30
        max_thresh2=200
        kernel_size=(1,31,2)
        exampleImg_sobelMag = mag_thresh(exampleImg_unwarp, kernel_size, (min_thresh2, max_thresh2))

        #--------------------------------------------------------------
        min_thresh3=0
        max_thresh3=0.09
        kernel_size2=7
        exampleImg_sobelDir = dir_thresh(exampleImg_unwarp, kernel_size2, (min_thresh3, max_thresh3))

        combined = np.zeros_like(exampleImg_sobelMag)
        combined[((exampleImg_sobelMag == 1) & (exampleImg_sobelDir == 1))] = 1

        exampleImg_SThresh = hls_sthresh(exampleImg_unwarp)
        exampleImg_LThresh = hls_lthresh(exampleImg_unwarp)
        exampleImg_LBThresh = lab_bthresh(exampleImg_unwarp)
        #-----------------------------------------------------------------
        combined_HLSl_LABb = np.zeros_like(exampleImg_LBThresh)
        combined_HLSl_LABb[((exampleImg_LBThresh == 1) | (exampleImg_LThresh == 1))] = 1

        combined_HLSs_LABb = np.zeros_like(exampleImg_SThresh)
        combined_HLSs_LABb[((exampleImg_LBThresh == 1) | (exampleImg_SThresh == 1))] = 1



        #cv2.imshow(img_arg+str(count)+".jpg", copy_image)
        #cv2.imshow(img_arg+str(count)+".jpg", resized_img)
        rows,cols,channels = imgOriginal.shape
        veca_slika=np.zeros_like(imgOriginal)

        small_org=cv2.resize(imgOriginal,(int(cols/3),int(rows/3)))
        red,stup,kanal = small_org.shape
        veca_slika[0:red,0:stup]=small_org

        resized_warped=cv2.resize(exampleImg_unwarp,(int(cols/3),int(rows/3)))
        red2,stup2,kanal2 = resized_warped.shape
        veca_slika[red:red+red2,0:stup]=resized_warped


        resized_warped=cv2.resize(exampleImg_undistort,(int(cols/3),int(rows/3)))
        red4,stup4,kanal4 = resized_warped.shape
        veca_slika[0:red,stup:stup+stup4]=resized_warped

        res=cv2.bitwise_and(exampleImg_unwarp,exampleImg_unwarp, mask=combined_HLSl_LABb)
        resized_warped_lines=cv2.resize(res,(int(cols/3),int(rows/3)))
        red3,stup3,kanal3 = resized_warped_lines.shape
        veca_slika[0:red,stup+stup2:cols-2]=resized_warped_lines

        binary_image=combined_HLSl_LABb
        histogram = np.sum(binary_image[binary_image.shape[0]//2:,:], axis=0)
        histogram_image=np.zeros((binary_image.shape[0]//2,binary_image.shape[1]),dtype=int)
        #histogram_image=np.ones((binary_image.shape[0]//2,binary_image.shape[1]),dtype=int)
        out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)
        i=1
        while i <= len(histogram)-1:

            if histogram[i]>0:
                #histogram_image[histogram_image.shape[0]-int(histogram[i]),i]=0
                cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(255,255,255),2)
            i+=1

        resizzeed=np.resize(out_image,(int(rows/3),int(cols/3)))

        # resized_histogram=np.zeros_like(resized_warped_lines)
        # resized_histogram[:,:,0]=resizzeed
        # resized_histogram[:,:,1]=resizzeed
        # resized_histogram[:,:,2]=resizzeed

        #print(resized_histogram.shape)
        #red2,stup2, kanal5 = resized_histogram.shape
        #veca_slika[0:red,stup+stup2:cols-2]=resized_histogram



        #cv2.imshow('histogram', out_image)
        cv2.imshow('histogram', resizzeed)

        cv2.imshow(img_arg+str(count)+".jpg", veca_slika)

        k = cv2.waitKey()
        if k == 83:
            count=count+1
        elif k == 81:
            if count !=0:
                count=count-1# hold windows open until user presses a key
        cv2.destroyAllWindows()                     # remove windows from memory
    cv2.destroyAllWindows()
    return

###################################################################################################
if __name__ == "__main__":
    main()
