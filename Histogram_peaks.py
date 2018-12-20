import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import Image_processing_functions as IPF
from scipy import signal


def draw_histogram(img_bin):
    #histogram image (middle right)
    histogram = np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)
    histogram_image=np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)
    #histogram_image=np.ones((binary_image.shape[0]//2,binary_image.shape[1]),dtype=int)
    out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)
    i=1
    while i <= len(histogram)-1:
        #histogram_image[histogram_image.shape[0]-int(histogram[i]),i]=0
        cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(255,255,255),2)
        i+=1
    cv2.putText(out_image, "4. Histogram image", (40,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
    return out_image

def find_histogram_peaks(histogram,histogram_image, image=False):

    out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)
    cv2.putText(out_image, "Histogram image", (40,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)

    peak_indices=[]
    i=1
    while i <= len(histogram)-2:

        cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(255,255,255),1)

        if(histogram[i]>=histogram[i-1] and histogram[i]>histogram[i+1]):

            peak_indices.append(i)
            #draw all found peaks
            #cv2.circle(out_image,(i,histogram_image.shape[0]-int(histogram[i])),2,(255,0,255),2)
        i+=1

    # print('before: ')
    # print(peak_indices)
    # print(histogram[peak_indices])

    index = 0
    while index <= len(peak_indices)-1:
        if index is 0:
            if (histogram[peak_indices[index]]<20):
                del peak_indices[index]
                index-=1

        elif (histogram[peak_indices[index]]>20):
            if (peak_indices[index]-peak_indices[index-1]<20):
                if(histogram[peak_indices[index-1]]<histogram[peak_indices[index]]):
                    keep_ind = index
                    del_ind=index-1
                else:
                    keep_ind = index-1
                    del_ind=index
                del peak_indices[del_ind]
                index-=1
        else:
            del peak_indices[index]
            index-=1
        index+=1

    #peak_count = len(peak_indices) # the number of peaks in the array
    i=0
    while i <= len(peak_indices)-1:
        cv2.circle(out_image,(peak_indices[i],histogram_image.shape[0]-int(histogram[peak_indices[i]])),2,(0,0,255),2)
        i+=1


    # print('after: ')
    # print(peak_indices)
    # print(histogram[peak_indices])
    if image is True:
        return peak_indices, out_image
    else:
        return peak_indices


def sliding_window_polyfit_all(img):
    # Take a histogram of the bottom half of the image
    #histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered

    #for using halves
    # leftx_base = np.argmax(histogram[:midpoint])
    # rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #for using middle quarters
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

    #leftx_base = np.argmax(histogram[:midpoint])
    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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
    margin = 100
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


    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data[0]





def main():
    dashcam_image_path = '/home/profesor/Documents/[ADAS]_Finding_Lanes/dashcam_driving/'
    img_arg="frame"
    count = 0
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
#--------------------------------------------------------------------------------------------------------------------
        #processing image and returning binary image

        new_img = np.copy(imgOriginal)

        #img_bin, Minv, img_unwarped = Lff.pipeline(new_img, diagnostic_images=True)
        img_bin, Minv, img_unwarped = IPF.pipeline(new_img)
#--------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
        #histogram image (middle right)
        #height,width=960,1280


        #out_image=draw_histogram(img_bin)

        # histogram = np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)
        # histogram_image=np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)
        #peaks,out_image=find_histogram_peaks(histogram,histogram_image,image=True)
        peaks,histogram_image=Lff.find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)

        print(peaks)
#------------------------------------------------------------------------------------------------------------------------------

        # histogram = np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)
        # histogram_image=np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)
        # #histogram_image=np.ones((binary_image.shape[0]//2,binary_image.shape[1]),dtype=int)
        # out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)
        # cv2.putText(out_image, "4. Histogram image", (40,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
        #
        #
        # peak_indices=[]
        # i=1
        # while i <= len(histogram)-2:
        #     #histogram_image[histogram_image.shape[0]-int(histogram[i]),i]=0
        #     cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(255,255,255),1)
        #
        #     #if((histogram[i]>histogram[i-1] & histogram[i]>histogram[i+1])and()):
        #     if(histogram[i]>=histogram[i-1] and histogram[i]>histogram[i+1]):
        #         #cv2.circle(out_image,(i,histogram_image.shape[0]-int(histogram[i])),2,(255,255,0),2)
        #         peak_indices.append(i)
        #     i+=1
        # print('before: ')
        # print(peak_indices)
        # print(histogram[peak_indices])
        #
        # # #y_coordinates = np.array(histogram) # convert your 1-D array to a numpy array if it's not, otherwise omit this line
        # # y_coordinates = np.copy(histogram) # convert your 1-D array to a numpy array if it's not, otherwise omit this line
        # # peak_widths = np.arange(1, 8)
        # # peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
        #
        #
        # peak_count = len(peak_indices) # the number of peaks in the array
        # i=0
        # while i <= peak_count-1:
        #     #cv2.circle(out_image,(peak_indices[i],histogram_image.shape[0]-int(histogram[peak_indices[i]])),2,(255,0,255),2)
        #     i+=1
        #
        #
        # index = 0
        # while index <= len(peak_indices)-1:
        #     if index is 0:
        #         if (histogram[peak_indices[index]]<20):
        #             del peak_indices[index]
        #             index-=1
        #
        #     elif (histogram[peak_indices[index]]>20):
        #         #print(histogram[peak_indices[index]])
        #         if (peak_indices[index]-peak_indices[index-1]<20):
        #             #max_indices, min_indices=max(peak_indices[index+1],peak_indices[index]),min(peak_indices[index+1],peak_indices[index])
        #             if(histogram[peak_indices[index-1]]<histogram[peak_indices[index]]):
        #                 keep_ind = index
        #                 del_ind=index-1
        #             else:
        #                 keep_ind = index-1
        #                 del_ind=index
        #             del peak_indices[del_ind]
        #             index-=1
        #             # print(max_indices)
        #             # print(min_indices)
        #             # print(' ')
        #     else:
        #         del peak_indices[index]
        #         index-=1
        #     index+=1
        #
        # peak_count = len(peak_indices) # the number of peaks in the array
        # i=0
        # while i <= peak_count-1:
        #     cv2.circle(out_image,(peak_indices[i],histogram_image.shape[0]-int(histogram[peak_indices[i]])),2,(0,0,255),2)
        #     i+=1
        #
        #
        # print('after: ')
        # print(peak_indices)
        # print(histogram[peak_indices])
#------------------------------------------------------------------------------------------------------------------------------
        #img_all_fits=cv2.resize(out_image,(int(width/3),int(height/4)))
#--------------------------------------------------------------------------------------------------------
        #draw new image with all find lanes as curves and best line that represents curves
        #img_all_fits = Lff.draw_all_curves(img_bin, l_line, r_line)
#---------------------------------------------------------------------------------------------------------
        #processed_image = new_img

#-------------------------------------------------------------------------------------
        #processed_image=np.copy(imgOriginal)
        #final_image = Lff.combine_images(imgOriginal,img_unwarped,img_bin,curves_image,img_all_fits,processed_image)
#----------------------------------------------------------------------------------------------------------------------
        #final_image = Lff.process_image(imgOriginal)

#--------------------------------------------------------------------------------------------------------------------
        cv2.imshow(img_arg+str(count)+".jpg", histogram_image)
        #print(final_image.shape)
        #cv2.imwrite('Output_'+img_arg+str(count)+".jpg",processed_image)
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
