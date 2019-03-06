import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import function_parameters as FP

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

def lab_threshold(img, thresh=(190,255), color='b'):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    if color == 'l':
        lab_b = lab[:,:,0]
    elif color == 'b':
        lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 175:
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(img[:,:,2])
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

#Function that thresholds one HLS channel and returns binary image
def hls_threshold(img, thresh=(125, 255), color='s'):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    if color == 's':
        binary_output = np.zeros_like(hls[:,:,2])
        binary_output[(hls[:,:,2] > thresh[0]) & (hls[:,:,2] <= thresh[1])] = 1
    elif color == 'l':
        hls_l = hls[:,:,1]
        hls_l = hls_l*(255/np.max(hls_l))
        binary_output = np.zeros_like(hls[:,:,2])
        binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    #elif color == 'h':

    else:
        return None
    # 3) Return a binary image of threshold result
    return binary_output

def binary_threshold(binary_img,thresh=1):

    thresh_binary_image=np.zeros_like(binary_img[:,:])
    thresh_binary_image[(binary_img[:,:] >= thresh)] = 1
    return thresh_binary_image

def compare_binary_images(ref_img,bin_img,diagnostic=False):
    ctr0=0
    ctr1=0
    ctr2=0
    ctr3=0
    ctr4=0
    ctr5=0
    ctr6=0
    ctr7=0
    ctr8=0
    ctr9=0
    ctr10=0

    for x in range(ref_img.shape[0]):
        for y in range(ref_img.shape[1]):
            if ref_img[x,y]==1:
                ctr0+=1
            if ref_img[x,y] == bin_img[0][x,y]:
                ctr1+=1
            else: ctr1-=1
            if ref_img[x,y]==bin_img[1][x,y]:
                ctr2+=1
            else: ctr2-=1
            if ref_img[x,y]==bin_img[2][x,y]:
                ctr3+=1
            else: ctr3-=1
            if ref_img[x,y]==bin_img[3][x,y]:
                ctr4+=1
            else: ctr4-=1
            if ref_img[x,y]==bin_img[4][x,y]:
                ctr5+=1
            else: ctr5-=1
            if ref_img[x,y]==bin_img[5][x,y]:
                ctr6+=1
            else: ctr6-=1
            if ref_img[x,y]==bin_img[6][x,y]:
                ctr7+=1
            else: ctr7-=1
            if ref_img[x,y]==bin_img[7][x,y]:
                ctr8+=1
            else: ctr8-=1
            if ref_img[x,y]==bin_img[8][x,y]:
                ctr9+=1
            else: ctr9-=1
            if ref_img[x,y]==bin_img[9][x,y]:
                ctr10+=1
            else: ctr10-=1

    razlika=[(ctr1,'RGB-R'), (ctr2,'RGB-G'),(ctr3,'RGB-B'),(ctr4,'Mag+Dir'),(ctr5,'HLS-L'),(ctr6,'HLS-S'),(ctr7,'Mag+Abs'),(ctr8,'Abs+Dir'),(ctr9,'LAB-B'),(ctr10,'LAB-L')]
    razlika.sort(key=lambda x: x[0], reverse=True)
    if diagnostic:
        print('REF image total 1 count='+str(ctr0))
        print('RGB R= '+str(ctr1))
        print('RGB G= '+str(ctr2))
        print('RGB B= '+str(ctr3))
        print('Mag+Dir= '+str(ctr4))
        print('HLS L= '+str(ctr5))
        print('HLS S= '+str(ctr6))
        print('Mag+Abs= '+str(ctr7))
        print('Abs+Dir= '+str(ctr8))
        print('LAB B= '+str(ctr9))
        print('LAB L= '+str(ctr10))


        print('Best combinations: '+razlika[0][1]+' OR '+razlika[1][1])

    return razlika


def compare_binary_images_NEW(ref_img,bin_img,lista,diagnostic=False):
    if(len(lista)!=0):
        counters=[]
        razlika=[]
        # counters.append(0)
        # kanter=0
        for x in bin_img:
            counters.append([0,0])


        for x in range(ref_img.shape[0]):
            for y in range(ref_img.shape[1]):
                for index,z in enumerate(counters):
                # for index in len(counters):
                    # print('ref_img['+str(x)+','+str(y)+']==bin_img['+str(index)+']['+str(x)+','+str(y)+']')

                    if ref_img[x,y] == bin_img[index][x,y]:
                        if ref_img[x,y]==1:
                            counters[index][0]+=1
                        else:
                            counters[index][1]+=1
                    else:
                        if ref_img[x,y]==1:
                            counters[index][0]-=1
                        else:
                            counters[index][1]-=1
                    # print(counters[index])
        # print('total white pixels: '+str(kanter))
        for index,x in enumerate(counters):
            print(str(counters[index]))
        print('')

        for index,x in enumerate(counters):
            prefix=1
            if counters[index][0]<0 or counters[index][1]<0:
                prefix=-1
            # print('prefix: '+str(prefix))
            pow1=(counters[index][0])**(2)
            pow2=(counters[index][1])**(2)
            rez=((pow1+pow2)**(0.5))*prefix
            razlika.append([rez,lista[index]])
            if diagnostic:
                print(str(lista[index])+' - '+str(rez))

        # razlika=[(counters[1],str(lista[0])), (counters[2],str(lista[1])),(counters[3],str(lista[2])),(counters[4],str(lista[3])),(counters[5],str(lista[4])),(counters[6],str(lista[5])),(counters[7],str(lista[6])),(counters[8],str(lista[7])),(counters[9],str(lista[8])),(counters[10],str(lista[9]))]

        # for index,x in enumerate(counters):
        #
        #     # pow1=int((x[index][0]))**(2)
        #     print(str(counters[index][1]))

        razlika.sort(key=lambda x: x[0], reverse=True)

        print('')
        print('sortirano:')
        for index,x in enumerate(razlika):
                # print(str(razlika[index])+' - '+str(x))
                print(str(razlika[index]))
        # print('Best combinations: '+razlika[0][1]+' OR '+razlika[1][1])

    return razlika

# def compare_binary_images_NEW(ref_img,bin_img,lista,diagnostic=False):
#     if(len(lista)!=0):
#         counters=[]
#         razlika=[]
#         # counters.append(0)
#         # kanter=0
#         for x in bin_img:
#             counters.append(0)
#
#         for x in range(ref_img.shape[0]):
#             for y in range(ref_img.shape[1]):
#                 # if ref_img[x,y]==1:
#                 #     kanter+=1
#                 for index,z in enumerate(counters):
#                     if ref_img[x,y] == bin_img[index][x,y]:
#                         counters[index]+=1
#                     else: counters[index]-=1
#         # print('total white pixels: '+str(kanter))
#         for index,x in enumerate(counters):
#             razlika.append([x,lista[index]])
#             if diagnostic:
#                 print(str(lista[index])+' - '+str(x))
#         # razlika=[(counters[1],str(lista[0])), (counters[2],str(lista[1])),(counters[3],str(lista[2])),(counters[4],str(lista[3])),(counters[5],str(lista[4])),(counters[6],str(lista[5])),(counters[7],str(lista[6])),(counters[8],str(lista[7])),(counters[9],str(lista[8])),(counters[10],str(lista[9]))]
#         razlika.sort(key=lambda x: x[0], reverse=True)
#
#         # print('Best combinations: '+razlika[0][1]+' OR '+razlika[1][1])
#
#     return razlika


def make_binary_stack(exampleImg_unwarp):

    min_thresh=25
    max_thresh=255
    exampleImg_sobelAbs = sobel_abs_thresh(exampleImg_unwarp, 'x', min_thresh, max_thresh)

    min_thresh2=25
    max_thresh2=255
    kernel_size=25
    exampleImg_sobelMag = sobel_mag_thresh(exampleImg_unwarp, kernel_size, (min_thresh2, max_thresh2))

    min_thresh3=0
    max_thresh3=0.11
    kernel_size2=7
    exampleImg_sobelDir = sobel_dir_thresh(exampleImg_unwarp, kernel_size2, (min_thresh3, max_thresh3))

    sobelMag_sobelDir = np.zeros_like(exampleImg_sobelMag)
    sobelMag_sobelDir[((exampleImg_sobelMag == 1) & (exampleImg_sobelDir == 1))] = 1

    sobelAbs_sobelDir = np.zeros_like(exampleImg_sobelAbs)
    sobelAbs_sobelDir[((exampleImg_sobelAbs == 1) & (exampleImg_sobelDir == 1))] = 1

    sobelMag_sobelAbs = np.zeros_like(exampleImg_sobelMag)
    sobelMag_sobelAbs[((exampleImg_sobelMag == 1) & (exampleImg_sobelAbs == 1))] = 1

    exampleImg_SThresh = hls_threshold(exampleImg_unwarp,thresh=(125,255),color='s')
    exampleImg_LThresh = hls_threshold(exampleImg_unwarp,thresh=(220,255),color='l')
    exampleImg_LBThresh = lab_threshold(exampleImg_unwarp,thresh=(190,255),color='b')
    exampleImg_LLBThresh = lab_threshold(exampleImg_unwarp,thresh=(190,255),color='l')
    exampleImg_RRGBThresh = rgb_thresh(exampleImg_unwarp,color=0)
    exampleImg_GRGBThresh = rgb_thresh(exampleImg_unwarp,color=1)
    exampleImg_BRGBThresh = rgb_thresh(exampleImg_unwarp,color=2)

    slike=[exampleImg_RRGBThresh,exampleImg_GRGBThresh,exampleImg_BRGBThresh,sobelMag_sobelDir,exampleImg_LThresh,exampleImg_SThresh,sobelMag_sobelAbs,sobelAbs_sobelDir,exampleImg_LBThresh,exampleImg_LLBThresh]
    #delete plot
    #added_binary_images=np.zeros_like(exampleImg_unwarp)
    added_binary_images=np.zeros_like(exampleImg_unwarp[:,:,0])
    #print(added_binary_images.dtype)
    added_binary_images=sobelMag_sobelAbs+sobelAbs_sobelDir+sobelMag_sobelDir+exampleImg_BRGBThresh+exampleImg_GRGBThresh+exampleImg_RRGBThresh+exampleImg_LLBThresh+exampleImg_LBThresh+exampleImg_LThresh+exampleImg_SThresh
    return added_binary_images,slike

def make_binary_stack_custom(exampleImg_unwarp,list):
    slike=[]
    tekst=[]
    added_binary_images=np.zeros_like(exampleImg_unwarp[:,:,0])
    kanali = split_channels(exampleImg_unwarp)
    if(len(list)!=0):
        for x in list:
            if str(x)=='rgb_r':
                exampleImg_RRGBThresh = rgb_thresh(exampleImg_unwarp,color=0)
                slike.append(exampleImg_RRGBThresh)
                tekst.append('rgb_r')
            if str(x)=='hls_s':
                exampleImg_SThresh = hls_threshold(exampleImg_unwarp,thresh=(125,255),color='s')
                slike.append(exampleImg_SThresh)
                tekst.append('hls_s')
            if str(x)=='hls_l':
                exampleImg_LThresh = hls_threshold(exampleImg_unwarp,thresh=(220,255),color='l')
                slike.append(exampleImg_LThresh)
                tekst.append('hls_l')
            if str(x)=='lab_l':
                exampleImg_LLBThresh = lab_threshold(exampleImg_unwarp,thresh=(190,255),color='l')
                slike.append(exampleImg_LLBThresh)
                tekst.append('lab_l')
            if str(x)=='lab_b':
                exampleImg_LBThresh = lab_threshold(exampleImg_unwarp,thresh=(190,255),color='b')
                slike.append(exampleImg_LBThresh)
                tekst.append('lab_b')
            if str(x)=='sobel_mag':
                min_thresh2=25
                max_thresh2=255
                kernel_size=25
                exampleImg_sobelMag = sobel_mag_thresh(exampleImg_unwarp, kernel_size, (min_thresh2, max_thresh2))
                slike.append(exampleImg_sobelMag)
                tekst.append('sobel_mag')
            if str(x)=='sobel_abs':
                min_thresh=25
                max_thresh=255
                exampleImg_sobelAbs = sobel_abs_thresh(exampleImg_unwarp, 'x', min_thresh, max_thresh)
                slike.append(exampleImg_sobelAbs)
                tekst.append('sobel_abs')
            if str(x)=='sobel_dir':
                min_thresh3=0
                max_thresh3=0.11
                kernel_size2=7
                exampleImg_sobelDir = sobel_dir_thresh(exampleImg_unwarp, kernel_size2, (min_thresh3, max_thresh3))
                slike.append(exampleImg_sobelDir)
                tekst.append('sobel_dir')
            if str(x)=='hsv_white':
                white_hsv_low  = np.array([ 0,   0,   160])
                white_hsv_high = np.array([ 255,  80, 255])
                image_HSV = cv2.cvtColor(exampleImg_unwarp,cv2.COLOR_RGB2HSV)
                # res1 = color_mask(image_HSV,white_hsv_low,white_hsv_high)
                res11=np.zeros_like(image_HSV[:,:,0])
                res1 = cv2.inRange(image_HSV, white_hsv_low, white_hsv_high)
                res11[(res1 > 0)] = 1
                slike.append(res11)
                tekst.append('hsv_white')
            if str(x)=='hsv_yellow':
                yellow_hsv_low  = np.array([ 0,  100,  100])
                yellow_hsv_high = np.array([ 80, 255, 255])
                image_HSV = cv2.cvtColor(exampleImg_unwarp,cv2.COLOR_RGB2HSV)
                # res_mask = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
                res11=np.zeros_like(image_HSV[:,:,0])
                res_mask = cv2.inRange(image_HSV, yellow_hsv_low, yellow_hsv_high)
                res11[(res_mask > 0)] = 1
                slike.append(res11)
                tekst.append('hsv_yellow')
            if str(x)=='white_tight':
                slike.append(kanali['white_tight'])
                tekst.append('white_thight')
            if str(x)=='white_loose':
                slike.append(kanali['white_loose'])
                tekst.append('white_loose')
            if str(x)=='yellow_edge_pos':
                slike.append(kanali['yellow_edge_pos'])
                tekst.append('yellow_edge_pos')
            if str(x)=='yellow_edge_neg':
                slike.append(kanali['yellow_edge_neg'])
                tekst.append('yellow_edge_neg')
            if str(x)=='yellow':
                slike.append(kanali['yellow'])
                tekst.append('yellow')
            if str(x)=='edge_pos':
                slike.append(kanali['edge_pos'])
                tekst.append('edge_pos')
            if str(x)=='edge_neg':
                slike.append(kanali['edge_neg'])
                tekst.append('edge_neg')
            if str(x)=='hls_sobel':
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
                slike.append(final_binary)
                tekst.append('hls_sobel')


    for x in slike:
        added_binary_images+=x.astype(np.uint8)

    return added_binary_images,slike


def make_binary_stack2(lista):

    added_binary_images=np.zeros_like(lista[0])
    for image in lista:
        added_binary_images+=image
    return added_binary_images

def threshold_binary_stack(added_binary_images,threshh=1):
    thresh_binary_image=np.zeros_like(added_binary_images[:,:])
    thresh_binary_image[(added_binary_images[:,:] >= threshh)] = 1
    return thresh_binary_image

def calibrate_IPF(img_unwarp):
    print("_______________________________________________________________")
    print('Calibrating image filters',end="", flush=True)
    stacked_binary_image,all_binary_images=make_binary_stack(img_unwarp)
    print('.',end="", flush=True)
    max_value=np.max(stacked_binary_image)
    print('.',end="", flush=True)
    thresholded_binary_image=threshold_binary_stack(stacked_binary_image,2)
    print('.')
    top_two=compare_binary_images(thresholded_binary_image,all_binary_images)
    print('Done!')
    print('Best combinations are: '+top_two[0][1]+', '+ top_two[1][1]+', '+ top_two[2][1])
    print("________________________________________________________________")
    return top_two[0][1],top_two[1][1]


def calibrate_IPF_NEW(img_unwarp):
    print("_______________________________________________________________")
    print('Calibrating image filters',end="", flush=True)
    lista=['rgb_r','hls_s','hls_l','lab_l','lab_b','sobel_mag','sobel_abs','sobel_dir','hsv_white','hsv_yellow','white_tight','white_loose','yellow_edge_pos','yellow_edge_neg','yellow','edge_pos','hls_sobel']
    stacked_binary_image,all_binary_images=make_binary_stack_custom(img_unwarp,lista)
    print('.',end="", flush=True)
    max_value=np.max(stacked_binary_image)
    print('.',end="", flush=True)
    thresholded_binary_image=threshold_binary_stack(stacked_binary_image,2)
    print('.')
    top_two=compare_binary_images_NEW(thresholded_binary_image,all_binary_images,lista,diagnostic=True)
    print('Done!')
    print('Best combinations are: '+top_two[0][1]+', '+ top_two[1][1]+', '+ top_two[2][1])
    print("________________________________________________________________")
    return top_two[0][1],top_two[1][1]

def threshold_equation(max_value):
    if max_value < 2:
        threshold=max_value
    elif max_value >= 2 and max_value < 6:
        threshold=2
    else:
        threshold=max_value-4
    return threshold

def calibrate_IPF_yellow_white(img_unwarp,sobel=False):
    print("_______________________________________________________________")
    print('Calibrating image filters',end="", flush=True)
    lista_white=['rgb_r','hls_s','hls_l','lab_l','hsv_white','white_tight','white_loose']
    lista_yellow=['lab_b','hsv_yellow','yellow']
    stacked_binary_image_white,all_binary_images_white=make_binary_stack_custom(img_unwarp,lista_white)
    stacked_binary_image_yellow,all_binary_images_yellow=make_binary_stack_custom(img_unwarp,lista_yellow)
    if sobel:
        lista_sobel=['sobel_mag','sobel_abs','sobel_dir','edge_pos','edge_neg','hls_sobel']
        stacked_binary_image_sobel,all_binary_images_sobel=make_binary_stack_custom(img_unwarp,lista_sobel)
        stacked_binary_image_white=stacked_binary_image_white+stacked_binary_image_sobel
        stacked_binary_image_yellow=stacked_binary_image_yellow+stacked_binary_image_sobel

    print('.',end="", flush=True)
    max_value_white=np.max(stacked_binary_image_white)
    max_value_yellow=np.max(stacked_binary_image_yellow)

    threshold_white=threshold_equation(max_value_white)
    threshold_yellow=threshold_equation(max_value_yellow)
    # print(threshold_yellow)
    # print(threshold_white)
    print('.',end="", flush=True)

    thresholded_binary_image_white=threshold_binary_stack(stacked_binary_image_white,threshold_white)
    thresholded_binary_image_yellow=threshold_binary_stack(stacked_binary_image_yellow,threshold_yellow)
    print('.')
    top_yellow=compare_binary_images_NEW(thresholded_binary_image_yellow,all_binary_images_yellow,lista_yellow,diagnostic=True)
    print('')
    top_white=compare_binary_images_NEW(thresholded_binary_image_white,all_binary_images_white,lista_white,diagnostic=True)
    print('Done!')
    print('Best combinations are(yellow): '+top_yellow[0][1]+', '+ top_yellow[1][1]+', '+ top_yellow[2][1])
    print('Best combinations are(white): '+top_white[0][1]+', '+ top_white[1][1]+', '+ top_white[2][1])
    print("________________________________________________________________")
    return top_white,top_yellow


# def unwarp_points(h,w):
#     # src = np.float32([(592,450),
#     #                       (692,450),
#     #                       (209,622),
#     #                       (1121,622)])
#     # src = np.float32([(585,455),
#     #                       (705,455),
#     #                       (190,720),
#     #                       (1130,720)])
#
#     #ispravan!
#     # src = np.float32([(550,430),
#     #                       (730,430),
#     #                       (200,622),
#     #                       (1080,622)])
#     # src = np.float32([(550,430),
#     #                       (730,430),
#     #                       (200,622),
#     #                       (1080,622)])
#
#     # src = np.float32([(575,464),
#     #                   (707,464),
#     #                   (258,682),
#     #                   (1049,682)])
#     # dst = np.float32([(450,0),
#     #                       (w-450,0),
#     #                       (450,h),
#     #                       (w-450,h)])
#
#     src,dst= FP.unwarp_points(h,w)
#
#     return src,dst


def pipeline(img):

    # Undistort
    img_undistort = undistort(img)

    #get points on image for perspective transform
    h,w = img.shape[:2]
    src,dst = FP.unwarp_points(h,w)

    # Perspective Transform
    img_unwarp, M, Minv = unwarp(img_undistort, src, dst)

    # img_unwarp[:, :, 0] = cv2.equalizeHist(img_unwarp[:, :, 0])
    # img_unwarp[:, :, 1] = cv2.equalizeHist(img_unwarp[:, :, 1])
    # img_unwarp[:, :, 2] = cv2.equalizeHist(img_unwarp[:, :, 2])


    img_unwarp_inverted=cv2.cvtColor(img_unwarp, cv2.COLOR_RGB2BGR)
    kanali = split_channels(img_unwarp_inverted)
    list=[]
    # Sobel Absolute (using default parameters)
    if 'rgb_r' in FP.binary_combinations:
        exampleImg_RRGBThresh = rgb_thresh(img_unwarp,color=0)
        list.append(exampleImg_RRGBThresh)
    if 'sobel_abs' in FP.binary_combinations:
        min_thresh=25
        max_thresh=255
        img_sobelAbs = sobel_abs_thresh(img_unwarp, 'x', min_thresh, max_thresh)
        list.append(img_sobelAbs)
    # Sobel Magnitude (using default parameters)
    if 'sobel_mag' in FP.binary_combinations:
        min_thresh2=25
        max_thresh2=255
        kernel_size=25
        img_sobelMag = sobel_mag_thresh(img_unwarp, kernel_size, (min_thresh2, max_thresh2))
        list.append(img_sobelMag)
    # Sobel Direction (using default parameters)
    if 'sobel_dir' in FP.binary_combinations:
        min_thresh3=0
        max_thresh3=0.11
        kernel_size2=7
        img_sobelDir = sobel_dir_thresh(img_unwarp, kernel_size2, (min_thresh3, max_thresh3))
        list.append(img_sobelDir)
    # HLS S-channel Threshold (using default parameters)
    #img_HLS_s_thresh = hls_threshold(img_unwarp, thresh=(220, 255), color='s')
    if 'hls_s' in FP.binary_combinations:
        img_HLS_s_thresh = hls_threshold(img_unwarp, thresh=(125, 255), color='s')
        list.append(img_HLS_s_thresh)
    # HLS L-channel Threshold (using default parameters)
    #img_LThresh = hls_lthresh(img_unwarp)
    #img_LThresh = hls_threshold(img_unwarp, thresh=(125, 255), color='l')
    if 'hls_l' in FP.binary_combinations:
        img_HLS_l_thresh = hls_threshold(img_unwarp, thresh=(220, 255), color='l')
        list.append(img_HLS_l_thresh)
    # Lab L-channel Threshold (using default parameters)
    #img_LLThresh = lab_lthresh(img_unwarp)
    if 'lab_l' in FP.binary_combinations:
        #img_LAB_l_thresh = lab_threshold(img_unwarp, thresh=(190,255), color='l')
        img_LAB_l_thresh = lab_threshold(img_unwarp_inverted, thresh=(190,255), color='l')
        list.append(img_LAB_l_thresh)
    # Lab B-channel Threshold (using default parameters)
    if 'lab_b' in FP.binary_combinations:
        # img_LAB_b_thresh = lab_threshold(img_unwarp, thresh=(190,255), color='b')
        img_LAB_b_thresh = lab_threshold(img_unwarp_inverted, thresh=(190,255), color='b')
        list.append(img_LAB_b_thresh)
    # if 'hsv_white' in FP.binary_combinations:
    #     white_hsv_low  = np.array([ 0,   0,   160])
    #     white_hsv_high = np.array([ 255,  80, 255])
    #     image_HSV = cv2.cvtColor(img_unwarp,cv2.COLOR_RGB2HSV)
    #     # image_HSV = cv2.cvtColor(img_unwarp_inverted,cv2.COLOR_RGB2HSV)
    #     res1=np.zeros_like(img_unwarp)
    #     res1 = cv2.inRange(image_HSV,white_hsv_low,white_hsv_high)
    #     list.append(res1)
    # if 'hsv_yellow' in FP.binary_combinations:
    #     yellow_hsv_low  = np.array([ 0,  100,  100])
    #     yellow_hsv_high = np.array([ 80, 255, 255])
    #     def color_mask(hsv,low,high):
    #     # Return mask from HSV
    #         mask = cv2.inRange(hsv, low, high)
    #         return mask
    #     image_HSV = cv2.cvtColor(img_unwarp,cv2.COLOR_RGB2HSV)
    #     res_mask = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    #     list.append(res_mask)

    if 'white_tight' in FP.binary_combinations:
        list.append(kanali['white_tight'])
    if 'white_loose' in FP.binary_combinations:
        list.append(kanali['white_loose'])
    if 'yellow_edge_pos' in FP.binary_combinations:
        list.append(kanali['yellow_edge_pos'])
    if 'yellow_edge_neg' in FP.binary_combinations:
        list.append(kanali['yellow_edge_neg'])
    if 'yellow' in FP.binary_combinations:
        list.append(kanali['yellow'])
    if 'edge_pos' in FP.binary_combinations:
        list.append(kanali['edge_pos'])
    if 'edge_neg' in FP.binary_combinations:
        list.append(kanali['edge_neg'])
    if 'hls_sobel' in FP.binary_combinations:
        s_thresh=(150, 255)
        l_thresh=(120,255)
        sx_thresh=(20, 100)
        hls = cv2.cvtColor(img_unwarp, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Apply Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # This will take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from the horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        # Apply Thresholding
        final_binary = np.zeros_like(s_channel)
        final_binary[np.logical_or((s_channel > s_thresh[0]) & (s_channel < s_thresh[1]) & (l_channel > l_thresh[0]) & (l_channel < l_thresh[1]) , (scaled_sobel > sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]))] = 1
        list.append(final_binary)


    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_unwarp[:,:,0])
    #combined[(img_LAB_l_thresh == 1) | (img_HLS_s_thresh == 1)] = 1


    if len(list) is 1:
        combined[(list[0] == 1)] = 1
    elif len(list) is 2:
        combined[(list[1] == 1)|(list[0] == 1)] = 1
    elif len(list) is 3:
        combined[(list[0] == 1)|(list[1] == 1)|(list[2] == 1)] = 1
    elif len(list) is 4:
        combined[(list[0] == 1)|(list[1] == 1)|(list[2] == 1)|(list[3] == 1) ] = 1
    elif len(list) is 5:
        combined[(list[0] == 1)|(list[1] == 1) |(list[2] == 1)|(list[3] == 1)|(list[4] == 1)] = 1
    elif len(list) is 6:
        combined[(list[0] == 1)|(list[1] == 1) |(list[2] == 1)|(list[3] == 1)|(list[4] == 1)|(list[5] == 1)] = 1
    elif len(list) is 7:
        combined[(list[0] == 1)|(list[1] == 1) |(list[2] == 1)|(list[3] == 1)|(list[4] == 1)|(list[5] == 1)|(list[6] == 1)] = 1
    elif len(list) is 8:
        combined[(list[0] == 1)|(list[1] == 1) |(list[2] == 1)|(list[3] == 1)|(list[4] == 1)|(list[5] == 1)|(list[6] == 1)|(list[7] == 1)] = 1
    elif len(list) is 9:
        combined[(list[0] == 1)|(list[1] == 1) |(list[2] == 1)|(list[3] == 1)|(list[4] == 1)|(list[5] == 1)|(list[6] == 1)|(list[7] == 1)|(list[8] == 1)] = 1
    elif len(list) is 10:
        combined[(list[0] == 1)|(list[1] == 1) |(list[2] == 1)|(list[3] == 1)|(list[4] == 1)|(list[5] == 1)|(list[6] == 1)|(list[7] == 1)|(list[8] == 1)|(list[9] == 1)] = 1

    #combined[(img_HLS_s_thresh == 1)|(img_LAB_b_thresh == 1) ] = 1


    #combined[(img_LAB_b_thresh == 1)] = 1
    #combined[(img_SThresh == 1) | (img_LLThresh == 1)] = 1


    return combined, Minv, img_unwarp


def pipeline_all_binImg_combined(img):

    # Undistort
    img_undistort = undistort(img)

    #get points on image for perspective transform
    h,w = img.shape[:2]
    src,dst = FP.unwarp_points(h,w)

    # Perspective Transform
    img_unwarp, M, Minv = unwarp(img_undistort, src, dst)

    combined=make_binary_stack(img_unwarp,2)

    return combined, Minv, img_unwarp

def split_channels(image) :
        """
        returns a total of 7 channels :
        4 edge channels : all color edges (including the signs), yellow edges (including the signs)
        3 color channels : yellow and white (2 different thresholds are used for white)
        """
        binary = {}

        # thresholding parameters for various color channels and Sobel x-gradients
        h_thresh=(15, 35)
        s_thresh=(75, 255) #s_thresh=(30, 255)
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

def pipeline2(img):

    # Undistort
    img_undistort = undistort(img)

    #get points on image for perspective transform
    h,w = img.shape[:2]
    src,dst = FP.unwarp_points(h,w)

    # Perspective Transform
    img_unwarp, M, Minv = unwarp(img_undistort, src, dst)
    img_unwarp_inverted=cv2.cvtColor(img_unwarp, cv2.COLOR_RGB2BGR)

    #img_unwarp = cv2.cvtColor(img_unwarp, cv2.COLOR_BGR2RGB)
    kanali = split_channels(cv2.cvtColor(img_unwarp, cv2.COLOR_BGR2RGB))

    res_binary_edge=np.zeros_like(kanali['edge_pos'])
    res_binary_edge = cv2.bitwise_or(kanali['edge_pos'], kanali['edge_neg'])
    res_binary_white=np.zeros_like(kanali['edge_pos'])
    res_binary_white = cv2.bitwise_or(kanali['white_tight'], kanali['white_loose'])
    res_binary_yellow=np.zeros_like(kanali['edge_pos'])
    res_binary_yellow = cv2.bitwise_or(kanali['yellow_edge_neg'], kanali['yellow_edge_pos'])
    res_binary_yellow_yellow = cv2.bitwise_and(res_binary_yellow, kanali['yellow'])

    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(img_unwarp[:,:,0])
    # combined2 = np.zeros_like(img_unwarp[:,:,0])
    #combined[((res_binary_edge == 1) & (res_binary_white == 1)) | ((res_binary_edge == 1) & (res_binary_yellow == 1))] = 1

    # combined[(res_binary_edge == 1) & ((res_binary_white == 1) | (kanali['yellow'] == 1))] = 1
    combined[(res_binary_edge == 1) & ((res_binary_white == 1) | (res_binary_yellow_yellow == 1))] = 1

    #combined=cv2.bitwise_or(res_binary_white,kanali['yellow'])
    #combined=kanali['yellow']


    #combined[(img_HLS_s_thresh == 1)|(img_LAB_b_thresh == 1) ] = 1


    #combined[(img_LAB_b_thresh == 1)] = 1
    #combined[(img_SThresh == 1) | (img_LLThresh == 1)] = 1


    return combined, Minv, img_unwarp
