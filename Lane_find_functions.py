import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Image_processing_functions as IPF
import copy
import function_parameters as FP
from math import *

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #number of detected pixels
        self.px_count = None

        self.peak=0
        self.peakdiffs=0
        #self.keep_lines=5
        self.keep_lines=10
        self.confidence=0
        self.previous_fit=[]

    def add_fit(self, fit, inds,lane_number, peak_count):
        # add a found fit to the line, up to n
        ignored_fit=False
        # self.previous_fit=self.best_fit

        if fit is not None:

            # if self.peak is 0:
            #     self.peak=peak_count
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
                self.peakdiffs=abs(peak_count-self.peak)
            if ((self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 100.) and len(self.current_fit) > 0) or not (fit[2]>=0 and -1<=fit[1]<=1): #or not (self.peak-30<=peak_count<=self.peak+30)
                #(fit[2]>=0 and -1<=fit[1]<=1)
                #or not (fit[2]>=0 and -0.2<=fit[1]<=0.2)
               # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                    #if  ((self.peak-30) <= peak_count <= (self.peak+30)):
                    self.detected = False

                    if self.confidence-10 >= 0:
                        self.confidence-= 10

                    if self.confidence <= 0:
                        self.reset_lane()
                    ignored_fit=True
                    # print('peak ignored!')
                    #self.peak=0
            #good fit.
            else:
                if self.confidence+20<=100:
                    self.confidence+=20
                else:
                    self.confidence=100

                self.previous_fit=self.best_fit

                self.peak=peak_count
                # print('peak added!')
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > self.keep_lines:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-self.keep_lines:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)
        return ignored_fit

    def reset_lane(self):
        self.best_fit=None
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.current_fit = []
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0,0,0], dtype='float')
        self.px_count = None
        self.peak=0
        self.confidence=0
        self.previous_fit=[]

l_line = Line()
ll_line = Line()
r_line = Line()
rr_line = Line()

temp_line=Line()

# temp_temp_line=Line()

lane_list=[l_line, ll_line, r_line, rr_line]

def plot_fit_onto_img(img, fit, plot_color):
    if fit is None:
        return img
    new_img = np.copy(img)
    h = new_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    # for 2nd order polynominal
    plotx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    # for 1st order polynominal
    # plotx = fit[1]*ploty + fit[2]
    pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
    cv2.polylines(new_img, np.int32([pts]), isClosed=False, color=plot_color, thickness=8)
    return new_img

def find_histogram_peaks(histogram,histogram_image, image=False):

    out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)

    h,w=histogram_image.shape
    if w==1280:
        font_size=2
        thickness=2
        position=(40,40)
    elif w==640:
        font_size=1
        thickness=1
        position=(20,40)
    else:
        font_size=2
        thickness=2
        position=(40,40)

    cv2.putText(out_image, "4. Histogram image with peaks", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,255,0), thickness, cv2.LINE_AA)

    noise_count=0
    peak_indices=[]
    i=1
    while i <= len(histogram)-2:
        if histogram[i]>70:
        # if histogram[i]==0:
            noise_count+=1
        cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(255,255,255),2)

        if(histogram[i]>=histogram[i-1] and histogram[i]>histogram[i+1]):

            peak_indices.append(i)
            #draw all found peaks
            #cv2.circle(out_image,(i,histogram_image.shape[0]-int(histogram[i])),2,(255,0,255),2)
        i+=1
    noise_percent=(noise_count/(len(histogram)-2))*100
    ones_percent=100-noise_percent
    print('noise_percent= '+str(noise_percent)+'%')
    print('ones_percent= '+str(ones_percent)+'%')

    if noise_percent >=45:
        peak_indices=[]
        cv2.putText(out_image, "Image too noisy!", (200,200), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,255,255), thickness, cv2.LINE_AA)

    # print('before: ')
    # print(peak_indices)
    # print(histogram[peak_indices])

    index = 0
    #width=100
    if w==1280 or w==1228: #1228 is for one video
        #original values for 1280x720
        width=190
        range=25
    elif w==640:
        #custom values for 640x360
        width=100
        range=20
    else:
        width=100
        range=20


    while index <= len(peak_indices)-1:
        if index is 0:
            if (histogram[peak_indices[index]]<range):
                del peak_indices[index]
                index-=1

        elif (histogram[peak_indices[index]]>range):
            if (peak_indices[index]-peak_indices[index-1]<width):
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


def correct_false_lines_abcoef():
    correction_index=None
    false_count=0
    best_false_count=0
    num_of_lanes=4

    for i, elements in enumerate(lane_list):
        if lane_list[i].best_fit is not None:
            false_count=0
            comparatora=lane_list[i].best_fit[0]
            comparatorb=lane_list[i].best_fit[1]
            derivation_comparator_a=2*lane_list[i].best_fit[0]*360
            # derivation_comparator_b=lane_list[i].best_fit[1]
            # print(str(i)+'=>i '+str(lane_list[i]))
            # print('comparatora='+str(comparatora))
            # print('comparatorb='+str(comparatorb))
            # print('derivation_comparator_a='+str(derivation_comparator_a)+' |  derivation_comparator_b='+str(derivation_comparator_b))
            print('')
            for j, elementsz in enumerate(lane_list):
                if i is j or lane_list[j].best_fit is None:
                    continue
                else:
                    # print(str(j)+'=>j '+str(lane_list[j]))
                    iteration_derivation_a=2*lane_list[j].best_fit[0]*360
                    # iteration_derivation_b=lane_list[j].best_fit[1]
                    # print('lejns[j][1]-1.5e-01'+' < '+'comparator'+' < '+'lejns[j][1]+1.5e-01')
                    # print(str(lane_list[j].best_fit[1]-1.5e-01)+'<'+str(comparator)+'<'+str(lane_list[j].best_fit[1]+1.5e-01))
                    # print('comparator derivation= '+str(derivation_comparator_a)+'X+'+str(derivation_comparator_b)+'  <---> list['+str(j)+'].best_fit derivation= '+str(iteration_derivation_a)+'X+'+str(iteration_derivation_b))
                    print('comparator angle= '+str(degrees(atan(derivation_comparator_a)))+'  <--->  '+str(degrees(atan(iteration_derivation_a)))+' =list['+str(j)+'].best_fit angle= ')
                    print('abs razlika stupnjeva= '+str(abs(degrees(atan(derivation_comparator_a)) - degrees(atan(iteration_derivation_a)))))
                    # print('abs(lane_list[j].best_fit[1] - comparatorb)='+str(lane_list[j].best_fit[1] - comparatorb)+'='+str(lane_list[j].best_fit[1]) +' - '+ str(comparatorb))
                    # print('abs(lane_list[j].best_fit[1] - comparatora)='+str(lane_list[j].best_fit[0] - comparatora)+'='+str(lane_list[j].best_fit[0]) +' - '+ str(comparatora))
                    # print(str(lane_list[j].best_fit[1]-1.5e-01)+'<'+str(comparator)+'<'+str(lane_list[j].best_fit[1]+1.5e-01))
                    # print('')
                    # if (lane_list[j].best_fit[1]-1.5e-01) < comparator < (lane_list[j].best_fit[1] + 1.5e-01):
                    # if abs(lane_list[j].best_fit[1] - comparatorb) < abs((lane_list[j].best_fit[1]*0.2+lane_list[j].best_fit[1])-lane_list[j].best_fit[1])0.31 and abs(abs(lane_list[j].best_fit[0]) - comparatora) < 0.00035 :
                    # if asb(lane_list[j].best_fit[1]-lane_list[j].best_fit[1]*0.5) < abs(comparatorb) < abs(lane_list[j].best_fit[1] + lane_list[j].best_fit[1]*0.5) and (lane_list[j].best_fit[0]-lane_list[j].best_fit[0]*0.5) < comparatora < (lane_list[j].best_fit[0] + lane_list[j].best_fit[0]*0.5) :
                    # if abs(lane_list[j].best_fit[1])-abs(lane_list[j].best_fit[1]) < abs(comparatorb) < abs(lane_list[j].best_fit[1]) + abs(lane_list[j].best_fit[1]):
                    # if abs(lane_list[j].best_fit[1]-lane_list[j].best_fit[1]*0.8) > abs(comparatorb) > abs(lane_list[j].best_fit[1] + lane_list[j].best_fit[1]*0.8) and  abs(lane_list[j].best_fit[0]-lane_list[j].best_fit[0]*0.8) < abs(comparatora) < abs(lane_list[j].best_fit[0] + lane_list[j].best_fit[0]*0.8):
                    if abs(degrees(atan(derivation_comparator_a)) - degrees(atan(iteration_derivation_a))) < 10: #mozda tocno
                        print('curves are similar')
                    else:
                        false_count+=1
                        print('curves are NOT similar')

            # if false_count>1 and false_count>best_false_count:
            if false_count>1 and false_count>best_false_count:
                best_false_count=false_count
                correction_index=i

            print('false_count='+str(false_count))
            print('best_false_count='+str(best_false_count))
            print('-----------------')
        else:
            num_of_lanes-=1
    print('correction_index='+str(correction_index))
    print('length of lane_list='+str(num_of_lanes))
    averagea=0
    averageb=0
    divider=0
    correctedBValue=None
    correctedAValue=None
    X=360
    if correction_index is not None and best_false_count is num_of_lanes-1:
        for i, elements in enumerate(lane_list):
            if correction_index is i or lane_list[i].best_fit is None :
                continue
            else:
                print('lejns['+str(i)+'][1]='+str(lane_list[i]))
                print('averagea='+str(averagea))
                print('averageb='+str(averageb))
                averagea+=lane_list[i].best_fit[0]
                averageb+=lane_list[i].best_fit[1]
                divider+=1
                print('averagea+lejns['+str(i)+'][0]='+str(averagea))
                print('averageb+lejns['+str(i)+'][1]='+str(averageb))
                print('divider='+str(divider))
        if divider is not 0:
            correctedBValue=averageb/divider
            correctedAValue=averagea/divider
        print('correctedAValue='+str(correctedAValue))
        print('correctedBValue='+str(correctedBValue))
        signbita = -1 if correctedAValue < 0 else 1
        signbitb = -1 if correctedBValue < 0 else 1
        print('signbita='+str(signbita))
        print('signbitb='+str(signbitb))
        c_coef=lane_list[correction_index].best_fit[0]*X*X+lane_list[correction_index].best_fit[1]*X+lane_list[correction_index].best_fit[2]
        print('c_coef='+str(c_coef))
        lane_list[correction_index].best_fit[0]=abs(correctedAValue)*signbita
        lane_list[correction_index].best_fit[1]=abs(correctedBValue)*signbitb
        lane_list[correction_index].best_fit[2]=c_coef
        print('correctedAValue w/sign='+str(correctedAValue))
        print('correctedBValue w/sign='+str(correctedBValue))

    if correctedBValue is None:
        return False,correction_index
    else:
        return True,correction_index





def find_4_histogram_peaks(histogram,histogram_image, image=False):

    out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)
    cv2.putText(out_image, "Histogram image with peaks", (40,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)

    peak_indices=[]
    i=1
    while i <= len(histogram)-2:

        cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(255,255,255),2)

        if(histogram[i]>=histogram[i-1] and histogram[i]>histogram[i+1]):

            peak_indices.append(i)
            #draw all found peaks
            #cv2.circle(out_image,(i,histogram_image.shape[0]-int(histogram[i])),2,(255,0,255),2)
        i+=1

    # print('before: ')
    # print(peak_indices)
    # print(histogram[peak_indices])

    index = 0
    #width=100
    width=190
    range=25

    while index <= len(peak_indices)-1:
        if index is 0:
            if (histogram[peak_indices[index]]<range):
                del peak_indices[index]
                index-=1

        elif (histogram[peak_indices[index]]>range):
            if (peak_indices[index]-peak_indices[index-1]<width):
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
    #print(peak_indices)
    four_peaks=[None,None,None,None]
    #width_margin=40

    for i in peak_indices:
        if i < 320:
            four_peaks[0] = i
        elif 320 < i < 640:
            four_peaks[1] = i
        elif 640< i < 930:
            four_peaks[2] = i
        elif 930 < i < 1280:
            four_peaks[3] = i


    cv2.line(out_image,(320,0),(320,720),(255,255,0),2)
    cv2.line(out_image,(640,0),(640,720),(255,255,0),2)
    cv2.line(out_image,(930,0),(930,720),(255,255,0),2)
    cv2.line(out_image,(1280,0),(1280,720),(255,255,0),2)

    peak_count = len(peak_indices) # the number of peaks in the array
    # i=0
    # while i <= len(peak_indices)-1:
    #     cv2.circle(out_image,(peak_indices[i],histogram_image.shape[0]-int(histogram[peak_indices[i]])),2,(0,0,255),2)
    #     i+=1

    i=0
    sirina=100
    while i < len(four_peaks):
        if four_peaks[i] is not None:
            cv2.circle(out_image,(four_peaks[i],histogram_image.shape[0]-int(histogram[four_peaks[i]])),2,(0,0,255),2)
            cv2.putText(out_image, str(four_peaks[i]), (four_peaks[i]-25,histogram_image.shape[0]-int(histogram[four_peaks[i]]+25)), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            #cv2.line(out_image,(four_peaks[i]+sirina,0),(four_peaks[i]+sirina,720),(0,0+80*i,255),3)
            #cv2.line(out_image,(four_peaks[i]-sirina,0),(four_peaks[i]-sirina,720),(0,0+80*i,255),3)

        i+=1



    # print('after: ')
    # print(peak_indices)
    # print(histogram[peak_indices])
    if image is True:
        #print(four_peaks)
        #print('')
        return four_peaks, out_image
    else:
        return four_peaks

def find_44_histogram_peaks(peak_indices, out_image,histogram):

    four_peaks=[None,None,None,None]
    width1_l=0
    width1_r=320
    width2_l=320
    width2_r=640
    width3_l=640
    width3_r=930
    width4_l=930
    width4_r=1280

    for i in peak_indices:
        if i < 320:
            four_peaks[0] = i
        elif 320 < i < 640:
            four_peaks[1] = i
        elif 640< i < 930:
            four_peaks[2] = i
        elif 930 < i < 1280:
            four_peaks[3] = i

    cv2.line(out_image,(320,0),(320,720),(255,255,0),2)
    cv2.line(out_image,(640,0),(640,720),(255,255,0),2)
    cv2.line(out_image,(930,0),(930,720),(255,255,0),2)
    cv2.line(out_image,(1280,0),(1280,720),(255,255,0),2)

    peak_count = len(peak_indices) # the number of peaks in the array
    # i=0
    # while i <= len(peak_indices)-1:
    #     cv2.circle(out_image,(peak_indices[i],histogram_image.shape[0]-int(histogram[peak_indices[i]])),2,(0,0,255),2)
    #     i+=1

    i=0
    sirina=100
    while i < len(four_peaks):
        if four_peaks[i] is not None:
            cv2.circle(out_image,(four_peaks[i],out_image.shape[0]-int(histogram[four_peaks[i]])),2,(0,0,255),2)
            cv2.putText(out_image, str(four_peaks[i]), (four_peaks[i]-25,out_image.shape[0]-int(histogram[four_peaks[i]]+25)), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.line(out_image,(four_peaks[i]+sirina,0),(four_peaks[i]+sirina,720),(0,0+80*i,255),3)
            cv2.line(out_image,(four_peaks[i]-sirina,0),(four_peaks[i]-sirina,720),(0,0+80*i,255),3)

        i+=1

    return four_peaks, out_image



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



def sliding_window_polyfit(img):
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

def polyfit_using_prev_fit(binary_warped, left_fit_prev, right_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) &
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) &
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)

    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds

def polyfit_using_prev_fit_all(binary_warped, left_fit_prev):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30
    # margin = 40
    # margin = 80
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) &
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    left_fit_new = None
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)

    return left_fit_new, left_lane_inds

# Method to determine radius of curvature and distance from lane center
# based on binary image, polynomial fit, and L and R lane pixel indices
def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Again, extract left and right line pixel positions
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds]
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters

    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
    return left_curverad, right_curverad, center_dist

def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


def highlight_road(original_img, binary_img, four_fits, Minv):
    new_img = np.copy(original_img)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image

    fit=[None,None,None,None]
    pts=[None,None,None,None]


    for i, fitt in enumerate(four_fits):
        if fitt is not None:
            fit[i] = four_fits[i][0]*ploty**2 + four_fits[i][1]*ploty + four_fits[i][2]
            if i is 0 or i is 2:
                pts[i] = np.array([np.transpose(np.vstack([fit[i], ploty]))])
            else:
                pts[i] = np.array([np.flipud(np.transpose(np.vstack([fit[i], ploty])))])

    if pts[0] is not None and pts[1] is not None:
        pts_road_left = np.hstack((pts[0], pts[1]))
        cv2.fillPoly(color_warp, np.int_([pts_road_left]), (0, 0,255))
    if pts[1] is not None and pts[2] is not None:
        pts_road_middle = np.hstack((pts[1], pts[2]))
        cv2.fillPoly(color_warp, np.int_([pts_road_middle]), (0,255, 0))
    if pts[2] is not None and pts[3] is not None:
        pts_road_right = np.hstack((pts[2], pts[3]))
        cv2.fillPoly(color_warp, np.int_([pts_road_right]), (0, 0,255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 2, 0)
    #result = cv2.add(new_img,newwarp)
    return result

def draw_lane_custom(original_img, binary_img, l_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None:
        return original_img
    # print(l_fit.shape)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))
    combined = np.dstack((warp_zero, warp_zero, warp_zero))

    # h,w,_ = original_img.shape
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
    # for 2nd order polynominal
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    # for 1st order polynominal
    # left_fitx = l_fit[1]*ploty + l_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    #pts = np.hstack((pts_left, pts_right))

    # create_validate_image(binary_img, l_fit,l_fit,l_fit,l_fit, Minv)

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=5)
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 2, 0)
    #result = cv2.add(new_img,newwarp)
    return result

def create_validate_image(binary_img, fit_indices, Minv,compare_image_path):


    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    height,width = warp_zero.shape
    # color_warp1 = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))
    # color_warp3 = np.dstack((warp_zero, warp_zero, warp_zero))
    # color_warp4 = np.dstack((warp_zero, warp_zero, warp_zero))
    combined = np.dstack((warp_zero, warp_zero, warp_zero))
    h,w = binary_img.shape
    inversed_fits=[]
    for x,element in enumerate(fit_indices):
        if element is not None:

            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
            # for 2nd order polynominal
            left_fitx = element[0]*ploty**2 + element[1]*ploty + element[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        #pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        #cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=5)
            cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=15)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
            # cv2.imwrite('newwarp'+str(x)+'.png',newwarp)

            lanex=[]
            laney=[]
            for i in range(0, height):
                for j in range(0, width):
                    # print('['+str(i)+']'+'['+str(j)+']= '+str(color_warp[i,j,0])+','+str(color_warp[i,j,1])+','+str(color_warp[i,j,2])+',')
                    # print(color_warp[i,j,1])
                    if newwarp[i,j,1] == 255:
                        # counter+=1
                        lanex.append(j)
                        laney.append(i)
                        # lane_inds.append(good_left_inds)
            lane_fit  = None
            # Fit a second order polynomial to each
            if len(lanex) != 0:
                lane_fit = np.polyfit(laney, lanex, 1)
                # print('lane_fit====='+str(lane_fit))

            ploty = np.linspace(0, height-1, height)
            # for 2nd order polynominal
            plotx = lane_fit[0]*ploty + lane_fit[1]
            # plotx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
            pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
            cv2.polylines(color_warp2, np.int32([pts]), isClosed=False, color=(x+1,x+1,x+1), thickness=15)
            # cv2.polylines(color_warp2, np.int32([pts]), isClosed=False, color=(255,255,255), thickness=15)
            inversed_fits.append(lane_fit)


    new_img2=cv2.imread(compare_image_path)

    image_by_rows=np.sum(new_img2, axis=1)
    y_coordinate=0
    print(image_by_rows)
    for i,element in enumerate(image_by_rows):
        if element[0] != 0:
            y_coordinate=i
            break
    print('y coordinate: '+str(y_coordinate))

    color_warp2[0:y_coordinate,:,:]=0

    # cv2.imwrite('kombinirano.png',combined)
    combined=cv2.addWeighted(new_img2, 1, color_warp2, 2, 0)

    # cv2.imwrite('lanes_inv_transformed.png',color_warp2)

    return color_warp2,combined
# -------------------------------------------------------------
# def create_validate_image(binary_img, fit_indices, Minv):
#
#
#     # Create an image to draw the lines on
#     warp_zero = np.zeros_like(binary_img).astype(np.uint8)
#     height,width = warp_zero.shape
#     color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#     color_warp1 = np.dstack((warp_zero, warp_zero, warp_zero))
#     color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))
#     color_warp3 = np.dstack((warp_zero, warp_zero, warp_zero))
#     color_warp4 = np.dstack((warp_zero, warp_zero, warp_zero))
#     combined = np.dstack((warp_zero, warp_zero, warp_zero))
#     h,w = binary_img.shape
#
#     for x,element in enumerate(fit_indices):
#     if l_fit is not None:
#
#         ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
#         # for 2nd order polynominal
#         left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
#
#     # Recast the x and y points into usable format for cv2.fillPoly()
#         pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#     #pts = np.hstack((pts_left, pts_right))
#
#     # Draw the lane onto the warped blank image
#     #cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#     # cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=5)
#         cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=15)
#
#     # Warp the blank back to original image space using inverse perspective matrix (Minv)
#         newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
#     # cv2.imwrite('newwarp.png',newwarp)
#
#         lanex=[]
#         laney=[]
#         for i in range(0, height):
#             for j in range(0, width):
#                 # print('['+str(i)+']'+'['+str(j)+']= '+str(color_warp[i,j,0])+','+str(color_warp[i,j,1])+','+str(color_warp[i,j,2])+',')
#                 # print(color_warp[i,j,1])
#                 if newwarp[i,j,1] == 255:
#                     # counter+=1
#                     lanex.append(j)
#                     laney.append(i)
#                     # lane_inds.append(good_left_inds)
#         lane_fit  = None
#         # Fit a second order polynomial to each
#         if len(lanex) != 0:
#             lane_fit = np.polyfit(laney, lanex, 1)
#             print('lane_fit====='+str(lane_fit))
#
#         ploty = np.linspace(0, height-1, height)
#         # for 2nd order polynominal
#         plotx = lane_fit[0]*ploty + lane_fit[1]
#         # plotx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
#         pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
#         cv2.polylines(color_warp2, np.int32([pts]), isClosed=False, color=(255,255,255), thickness=8)
#
#
#
#     new_img2=cv2.imread("01350.png")
#     combined=cv2.addWeighted(new_img2, 1, color_warp2, 2, 0)
#     cv2.imwrite('kombinirano.png',combined)
#
#     cv2.imwrite('lanes_inv_transformed.png',color_warp2)
# # -------------------------------------------------------------



def draw_data(original_img, curv_rad, center_dist):
    new_img = np.copy(original_img)
    h = new_img.shape[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(curv_rad) + 'm'
    cv2.putText(new_img, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(new_img, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return new_img

def process_image(imgOriginal):

    #processing image and returning binary image

    new_img = np.copy(imgOriginal)

    #img_bin, Minv, img_unwarped = Lff.pipeline(new_img, diagnostic_images=True)
    img_bin, Minv, img_unwarped = IPF.pipeline(new_img)
    peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)

#--------------------------------------------------------------------------------
    #processing binary image to find lanes as curves
    height,width,_ = new_img.shape

    # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not l_line.detected or not r_line.detected:
        l_fit, r_fit, l_lane_inds, r_lane_inds, rectangles = sliding_window_polyfit(img_bin)
        curves_image = create_image_of_sliding_windows_polyfit(img_bin, l_fit, r_fit, l_lane_inds, r_lane_inds, rectangles)
    else:
        l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)
        curves_image = create_image_of_polyfit_using_prev_fit(img_bin, l_fit, r_fit, l_lane_inds, r_lane_inds)


#----------------------------------------------------------------------------------------
    # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
    if l_fit is not None and r_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        #h = img.shape[0]
        h=height
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        x_int_diff = abs(r_fit_x_int-l_fit_x_int)
        if abs(350 - x_int_diff) > 100:
            l_fit = None
            r_fit = None

    l_line.add_fit(l_fit, l_lane_inds)
    r_line.add_fit(r_fit, r_lane_inds)
#--------------------------------------------------------------------------------------------------------
    #draw new image with all find lanes as curves and best line that represents curves
    img_all_fits = draw_all_curves(img_bin, l_line, r_line)
#---------------------------------------------------------------------------------------------------------
    # draw the current best fit if it exists
    if l_line.best_fit is not None and r_line.best_fit is not None:
        img_out1 = draw_lane(imgOriginal, img_bin, l_line.best_fit, r_line.best_fit, Minv)
        rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                               l_lane_inds, r_lane_inds)
        processed_image = draw_data(img_out1, (rad_l+rad_r)/2, d_center)
    else:
        processed_image = new_img

#-------------------------------------------------------------------------------------
    #processed_image=np.copy(imgOriginal)
    final_image = combine_images(imgOriginal,img_unwarped,img_bin,histogram_image,curves_image,img_all_fits,processed_image)

    return final_image

def process_image_smaller(imgOriginal):

    #processing image and returning binary image

    new_img = np.copy(imgOriginal)

    #img_bin, Minv, img_unwarped = Lff.pipeline(new_img, diagnostic_images=True)
    img_bin, Minv, img_unwarped = IPF.pipeline(new_img)
    peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)

#--------------------------------------------------------------------------------
    rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
    lista=[]
    for peak in peaks:

        l_fit, l_lane_inds, rectangles = sliding_window_polyfit_all(img_bin,peak)
        #print(l_fit)
        #print(l_fit.shape)
        #if l_fit[0] < 0 :
        if l_fit[2] >= 0 and -1 <= l_fit[1] <= 1:
            #print('added')
            #print(' ')
            lista.append(l_fit)
            rectangle_img = create_image_of_sliding_windows_polyfit(rectangle_img,img_bin, l_fit, l_lane_inds, rectangles,colour=(255,255,0))
    #print('_____________________________________________________')
    img_out1=np.copy(imgOriginal)
    for elements in lista:
        img_out1 = draw_lane_custom(img_out1, img_bin, elements, Minv)
#-------------------------------------------------------------------------------------
    #processed_image=np.copy(imgOriginal)
    final_image = combine_images_smaller(img_out1,img_unwarped,img_bin,histogram_image,rectangle_img)

    return final_image

def allocate_peaks_to_4lanes(peaks):
    if len(peaks)<5:
        four_lanes=[None,None,None,None]
        lane_change=False

        for peak in peaks:

            if peak < 320:
                if four_lanes[0] is not None:
                    if four_lanes[1] is not None:
                        four_lanes[2]=peak
                    else:
                        four_lanes[1]=peak
                    lane_change=True
                else:
                    four_lanes[0] = peak

            elif  320 <= peak < 640:
                if four_lanes[1] is not None:
                    if four_lanes[2] is not None:
                        four_lanes[3]=peak
                    else:
                        four_lanes[2]=peak
                    lane_change=True
                else:
                    four_lanes[1] = peak

            elif  640<= peak < 936:
                if four_lanes[2] is not None:
                    if four_lanes[1] is None:
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=peak
                        lane_change=True

                    elif four_lanes[0] is None:
                        four_lanes[0]=four_lanes[1]
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=peak
                        lane_change=True
                    else:
                        four_lanes[3]=peak
                    #lane_change=True
                else:
                    four_lanes[2] = peak

            elif 936 <= peak < 1280:
                if four_lanes[3] is not None:
                    if four_lanes[2] is None:
                        four_lanes[2]=four_lanes[3]
                        four_lanes[3]=peak
                        lane_change=True
                    elif four_lanes[1] is None:
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=four_lanes[3]
                        four_lanes[3]=peak
                        lane_change=True
                    elif four_lanes[0] is None:
                        four_lanes[0]=four_lanes[1]
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=four_lanes[3]
                        four_lanes[3]=peak
                        lane_change=True
                    #lane_change=True
                    # else:
                    #     four_lanes[3]=peak
                else:
                    four_lanes[3] = peak
    else:
        print('Calculated more than 4 peaks: ' + str(len(peaks)) + '____________________________________________________________________________')
        # four_lanes=[]
        four_lanes=[None,None,None,None]
        lane_change=False
        # four_lanes=None
        # lane_change=None
    return four_lanes, lane_change


def sort_peaks_in_list(peaks):
    if len(peaks)<5:
        # four_lanes=[0,0,0,0]
        four_lanes=[None,None,None,None]
        for peak in peaks:

            if peak is None:
                continue
            elif peak < 320:
                if four_lanes[0] is not None:
                    if four_lanes[1] is not None:
                        four_lanes[2]=peak
                    else:
                        four_lanes[1]=peak
                else:
                    four_lanes[0] = peak

            elif  320 <= peak < 640:
                if four_lanes[1] is not None:
                    if four_lanes[2] is not None:
                        four_lanes[3]=peak
                    else:
                        four_lanes[2]=peak
                else:
                    four_lanes[1] = peak

            elif  640<= peak < 936:
                if four_lanes[2] is not None:
                    if four_lanes[1] is None:
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=peak

                    elif four_lanes[0] is None:
                        four_lanes[0]=four_lanes[1]
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=peak
                    else:
                        four_lanes[3]=peak
                    #lane_change=True
                else:
                    four_lanes[2] = peak

            elif 936 <= peak < 1280:
                if four_lanes[3] is not None:
                    if four_lanes[2] is None:
                        four_lanes[2]=four_lanes[3]
                        four_lanes[3]=peak
                    elif four_lanes[1] is None:
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=four_lanes[3]
                        four_lanes[3]=peak
                    elif four_lanes[0] is None:
                        four_lanes[0]=four_lanes[1]
                        four_lanes[1]=four_lanes[2]
                        four_lanes[2]=four_lanes[3]
                        four_lanes[3]=peak
                else:
                    four_lanes[3] = peak
    else:
        print('Calculated more than 4 peaks: ' + str(len(peaks)) + '____________________________________________________________________________')
        four_lanes=[None,None,None,None]
    return four_lanes

def switch_lanes(four_lanes):
    global temp_line
    global lane_list
    margin=60
    for i, peak in enumerate(four_lanes):
        if peak is not None:
            #print('lane_list['+str(i)+']='+str(lane_list[i].peak))
            #print('peak: '+str(peak))
            if i==0:
                if lane_list[1].peak-margin<=peak<=lane_list[1].peak+margin and lane_list[1].peak is not 0:
                    temp_line=copy.deepcopy(lane_list[0])

                    lane_list[0]=copy.deepcopy(lane_list[1])
                    lane_list[1]=copy.deepcopy(lane_list[2])
                    lane_list[2]=copy.deepcopy(lane_list[3])
                    lane_list[3].reset_lane()
                    print('lanes changed:0')
                    break
            elif i==1:
                if (lane_list[0].peak-margin<=peak<=lane_list[0].peak+margin) and lane_list[0].peak is not 0:
                    temp_line=copy.deepcopy(lane_list[3])

                    lane_list[3]=copy.deepcopy(lane_list[2])
                    lane_list[2]=copy.deepcopy(lane_list[1])
                    if four_lanes[1] is not None:
                        lane_list[1]=copy.deepcopy(lane_list[0])
                        lane_list[0].reset_lane()
                    else:
                        lane_list[1].reset_lane()
                    print('lanes changed:11')
                    print(temp_line.peak)
                    print('')
                    # temp_temp_line=copy.deepcopy(tempp_line)
                    break
                elif (lane_list[2].peak-margin<=peak<=lane_list[2].peak+margin)and lane_list[2].peak is not 0:
                    temp_line=copy.deepcopy(lane_list[0])

                    lane_list[0]=copy.deepcopy(lane_list[1])
                    lane_list[1]=copy.deepcopy(lane_list[2])
                    if four_lanes[2] is not None:
                        lane_list[2]=copy.deepcopy(lane_list[3])
                        lane_list[3].reset_lane()
                    else:
                        lane_list[2].reset_lane()
                    print('lanes changed:12')
                    break
            elif i==2:
                if (lane_list[1].peak-margin<=peak<=lane_list[1].peak+margin)and lane_list[1].peak is not 0:
                    temp_line=copy.deepcopy(lane_list[3])

                    lane_list[3]=copy.deepcopy(lane_list[2])
                    lane_list[2]=copy.deepcopy(lane_list[1])
                    if four_lanes[1] is not None:
                        lane_list[1]=copy.deepcopy(lane_list[0])
                        lane_list[0].reset_lane()
                    else:
                        lane_list[1].reset_lane()

                    print('lanes changed:21')
                    break
                elif (lane_list[3].peak-margin<=peak<=lane_list[3].peak+margin)and lane_list[3].peak is not 0:
                    temp_line=copy.deepcopy(lane_list[0])

                    lane_list[0]=copy.deepcopy(lane_list[1])
                    lane_list[1]=copy.deepcopy(lane_list[2])
                    if four_lanes[2] is not None:
                        lane_list[2]=copy.deepcopy(lane_list[3])
                        lane_list[3].reset_lane()
                    else:
                        lane_list[2].reset_lane()

                    print('lanes changed:22')
                    break
            elif i==3:
                if (lane_list[2].peak-margin<=peak<=lane_list[2].peak+margin)and lane_list[2].peak is not 0:
                    temp_line=copy.deepcopy(lane_list[3])
                    lane_list[3]=copy.deepcopy(lane_list[2])
                    lane_list[2]=copy.deepcopy(lane_list[1])
                    lane_list[1]=copy.deepcopy(lane_list[0])
                    lane_list[0].reset_lane()
                    print('lanes changed:3')
                    break


def sort_peaks_in_list_new(four_peaks,four_lanes_peaks_sorted):
    margin=90
    # new_four_lanes=[None,None,None,None]
    # sorted_four_peaks=[0,0,0,0]
    sorted_four_peaks=[None,None,None,None]
    # sorted_four_peaks=copy.deepcopy(four_peaks)
    # new_four_lanes=[x if x else 0 for x in four_lanes]
    print('[Before for] sorted_four_peaks: '+str(sorted_four_peaks))
    if four_lanes_peaks_sorted[0] is 0 and four_lanes_peaks_sorted[1] is 0 and four_lanes_peaks_sorted[2] is 0 and four_lanes_peaks_sorted[3] is 0:
        sorted_four_peaks=sort_peaks_in_list(four_peaks)
    else:
        for i, peak in enumerate(four_lanes_peaks_sorted):
            # print('if peak is not None or 0, '+str(peak))
            if peak is not 0:
                for j, peakk in enumerate(four_peaks):
                    if peakk is not None:
                        if peak-margin<=peakk<=peak+margin:
                            sorted_four_peaks[i]=peakk
            else:
                # print('entered in else')
                sorted_four_peaks[i]=four_peaks[i]

    return sorted_four_peaks



# def switch_lanes(four_lanes):
#     margin=60
#     for i, peak in enumerate(four_lanes):
#         if peak is not None:
#             #print('lane_list['+str(i)+']='+str(lane_list[i].peak))
#             #print('peak: '+str(peak))
#             if i==0:
#                 if lane_list[1].peak-margin<=peak<=lane_list[1].peak+margin and lane_list[1].peak is not 0:
#                     lane_list[0]=copy.deepcopy(lane_list[1])
#                     lane_list[1]=copy.deepcopy(lane_list[2])
#                     lane_list[2]=copy.deepcopy(lane_list[3])
#                     lane_list[3].reset_lane()
#                     print('lanes changed:0')
#                     break
#             elif i==1:
#                 if (lane_list[0].peak-margin<=peak<=lane_list[0].peak+margin) and lane_list[0].peak is not 0:
#                     lane_list[3]=copy.deepcopy(lane_list[2])
#                     lane_list[2]=copy.deepcopy(lane_list[1])
#                     lane_list[1]=copy.deepcopy(lane_list[0])
#                     lane_list[0].reset_lane()
#                     print('lanes changed:11')
#                     break
#                 elif (lane_list[2].peak-margin<=peak<=lane_list[2].peak+margin)and lane_list[2].peak is not 0:
#                     lane_list[0]=copy.deepcopy(lane_list[1])
#                     lane_list[1]=copy.deepcopy(lane_list[2])
#                     if four_lanes[2] is not None:
#                         lane_list[2]=copy.deepcopy(lane_list[3])
#                         lane_list[3].reset_lane()
#                     else:
#                         lane_list[2].reset_lane()
#                     print('lanes changed:12')
#                     break
#             elif i==2:
#                 if (lane_list[1].peak-margin<=peak<=lane_list[1].peak+margin)and lane_list[1].peak is not 0:
#                     lane_list[3]=copy.deepcopy(lane_list[2])
#                     lane_list[2]=copy.deepcopy(lane_list[1])
#                     if four_lanes[1] is not None:
#                         lane_list[1]=copy.deepcopy(lane_list[0])
#                         lane_list[0].reset_lane()
#                     else:
#                         lane_list[1].reset_lane()
#
#                     print('lanes changed:21')
#                     break
#                 elif (lane_list[3].peak-margin<=peak<=lane_list[3].peak+margin)and lane_list[3].peak is not 0:
#                     lane_list[0]=copy.deepcopy(lane_list[1])
#                     lane_list[1]=copy.deepcopy(lane_list[2])
#                     lane_list[2]=copy.deepcopy(lane_list[3])
#                     lane_list[3].reset_lane()
#                     print('lanes changed:22')
#                     break
#             elif i==3:
#                 if (lane_list[2].peak-margin<=peak<=lane_list[2].peak+margin)and lane_list[2].peak is not 0:
#                     lane_list[3]=copy.deepcopy(lane_list[2])
#                     lane_list[2]=copy.deepcopy(lane_list[1])
#                     lane_list[1]=copy.deepcopy(lane_list[0])
#                     lane_list[0].reset_lane()
#                     print('lanes changed:3')
#                     break

def reverse_switch_lanes(four_lanes_before):
    # temp_lane_list=[copy.deepcopy(lane_list[0]),copy.deepcopy(lane_list[1]),copy.deepcopy(lane_list[2]),copy.deepcopy(lane_list[3])]
    temp_lane_list=[Line(),Line(),Line(),Line()]
    global temp_line
    for i,peak in enumerate(four_lanes_before):
        # print('i = '+str(i))
        # print('peak = '+str(peak))
        # print('lane_list['+str(i)+'] = '+str(lane_list[i].peak))
        # print(temp_line.peak)
        # print('/-/-/-/-/-/-/-/-/-/')
        if peak is lane_list[0].peak:
            if peak is 0:
                temp_lane_list[i].reset_lane()
            else:
                temp_lane_list[i]=copy.deepcopy(lane_list[0])
        elif peak is lane_list[1].peak:
            if peak is 0:
                temp_lane_list[i].reset_lane()
            else:
                temp_lane_list[i]=copy.deepcopy(lane_list[1])
        elif peak is lane_list[2].peak:
            if peak is 0:
                temp_lane_list[i].reset_lane()
            else:
                temp_lane_list[i]=copy.deepcopy(lane_list[2])
        elif peak is lane_list[3].peak:
            if peak is 0:
                temp_lane_list[i].reset_lane()
            else:
                temp_lane_list[i]=copy.deepcopy(lane_list[3])
        else:
            if peak is not 0:
                # print('temp_line:added')
                # print(temp_line.peak)
                temp_lane_list[i]=copy.deepcopy(temp_line)
                temp_line.reset_lane()
            else:
                temp_lane_list[i].reset_lane()
                temp_line.reset_lane()




    # print('reverse_Switch_lanes')
    # print(temp_lane_list[0].peak)
    # print(temp_lane_list[1].peak)
    # print(temp_lane_list[2].peak)
    # print(temp_lane_list[3].peak)

    lane_list[0]=copy.deepcopy(temp_lane_list[0])
    lane_list[1]=copy.deepcopy(temp_lane_list[1])
    lane_list[2]=copy.deepcopy(temp_lane_list[2])
    lane_list[3]=copy.deepcopy(temp_lane_list[3])


    # print('lane_list peaks:')
    # print(lane_list[0].peak)
    # print(lane_list[1].peak)
    # print(lane_list[2].peak)
    # print(lane_list[3].peak)

def follow_other_lane(lane_number):
    lane_diff=[]
    all_diff=0
    for i, object in enumerate(lane_list):
        if object.previous_fit is not None and object.best_fit is not None and i is not lane_number:
            temp_diff=lane_list[i].best_fit[2]-lane_list[i].previous_fit[2]
            lane_list[i].previous_fit=lane_list[i].best_fit
            if temp_diff is not 0:
                lane_diff.append(temp_diff)

    # for i, diff in enumerate(lane_diff):
    print('Follow_Other_Lane activated -> lane number: '+str(lane_number)+' ->lane_diff length='+str(len(lane_diff)))
    if len(lane_diff) is not 0:
        for i in range(len(lane_diff)):
            print('lane_diff['+str(i)+']= '+str(lane_diff[i]))
            all_diff+=lane_diff[i]

        all_diff=all_diff/len(lane_diff)
        print('all_diff'+str(all_diff))
    return all_diff

    # print('lane_diff[1]= '+str(lane_diff[1]))
    # print('lane_diff[2]= '+str(lane_diff[2]))
    # print('lane_diff[3]= '+str(lane_diff[3]))

def check_lane_order():
    invalid_order=False
    no_none_list=[]
    print('lane_list.bestfit[]')
    for i,element in enumerate(lane_list):
        if element.best_fit is not None:
            no_none_list.append(element.best_fit)
    print('len(no_none_list)= '+str(len(no_none_list)))
    if len(no_none_list)>1:
        for i,element in enumerate(no_none_list):
            print(str(element))
            if i+1 < len(no_none_list):
                # if no_none_list[i]>no_none_list[i+1]:
                X=720
                c_coef_left=no_none_list[i][0]*X*X+no_none_list[i][1]*X+no_none_list[i][2]
                c_coef_right=no_none_list[i+1][0]*X*X+no_none_list[i+1][1]*X+no_none_list[i+1][2]
                if c_coef_left+100>=c_coef_right or no_none_list[i][2]+50>=no_none_list[i+1][2]:
                    invalid_order=True
    print('invalid_order= '+str(invalid_order))
    print('-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/')

    return invalid_order

# def check_lane_order():
#     invalid_order=False
#     no_none_list=[]
#     print('lane_list.bestfit[]')
#     for i,element in enumerate(lane_list):
#         if element.best_fit is not None:
#             no_none_list.append(element.best_fit[2])
#     print('len(no_none_list)= '+str(len(no_none_list)))
#     if len(no_none_list)>1:
#         for i,element in enumerate(no_none_list):
#             print(str(element))
#             if i+1 < len(no_none_list):
#                 # if no_none_list[i]>no_none_list[i+1]:
#                 if no_none_list[i]+20>=no_none_list[i+1]:
#                     invalid_order=True
#     print('invalid_order= '+str(invalid_order))
#     print('-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/')
#
#     return invalid_order

def update_lanes(img_bin,rectangle_img,four_lanes):
    # print('********************************************')
    # print('self.peaks=['+str(lane_list[0].peak)+' '+str(lane_list[1].peak)+' '+str(lane_list[2].peak)+' '+str(lane_list[3].peak)+']')
    # print('********************************************')
    four_lanes_before=[lane_list[0].peak,lane_list[1].peak,lane_list[2].peak,lane_list[3].peak]
    switch_lanes(four_lanes)
    # print('temp_line.peak after switch_lanes: '+str(temp_line.peak))
    # print('temp_temp_line.peak after switch_lanes: '+str(temp_temp_line.peak))
    four_lanes_after=[lane_list[0].peak,lane_list[1].peak,lane_list[2].peak,lane_list[3].peak]
    print(four_lanes_before[0])
    print(four_lanes_before[1])
    print(four_lanes_before[2])
    print(four_lanes_before[3])
    print('')
    print(four_lanes_after[0])
    print(four_lanes_after[1])
    print(four_lanes_after[2])
    print(four_lanes_after[3])
    print('**********************************')
    print('lanes detected:')
    print(str(lane_list[0].detected)+'///'+str(lane_list[1].detected)+'///'+str(lane_list[2].detected)+'///'+str(lane_list[3].detected))
    print('peaks of lines:')
    print(str(lane_list[0].peak)+'///'+str(lane_list[1].peak)+'///'+str(lane_list[2].peak)+'///'+str(lane_list[3].peak))
    print('best fit of lines:')
    print(str(lane_list[0].best_fit)+'///'+str(lane_list[1].best_fit)+'///'+str(lane_list[2].best_fit)+'///'+str(lane_list[3].best_fit))
    print('**********************************')
    # print('********************************************')
    # print('after switching lanes -> self.peaks=['+str(lane_list[0].peak)+' '+str(lane_list[1].peak)+' '+str(lane_list[2].peak)+' '+str(lane_list[3].peak)+']')
    # print('********************************************')
    # print('adding new peaks: ['+str(four_lanes[0])+' '+str(four_lanes[1])+' '+str(four_lanes[2])+' '+str(four_lanes[3])+']')
    # print('********************************************')

    # print('previous self.previous_fit[0]: '+str(lane_list[0].previous_fit))
    # print('previous self.previous_fit[1]: '+str(lane_list[1].previous_fit))
    # print('previous self.previous_fit[2]: '+str(lane_list[2].previous_fit))
    # print('previous self.previous_fit[3]: '+str(lane_list[3].previous_fit))
    print('?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?')
    print('[BEFORE] four_lanes= '+str(four_lanes))
    print('[BEFORE] four_lanes_after= '+str(four_lanes_after))
    print('-----------------------------------------------')
    four_lanes2 = sort_peaks_in_list_new(four_lanes,four_lanes_after)
    print('-----------------------------------------------')
    # four_lanes2=four_lanes_after
    print('[AFTER] four_lanes= '+str(four_lanes2))
    print('?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?')

    # four_lanes_after=(253, 528, 883, None)

    for i, peak in enumerate(four_lanes2):
        print('peak['+str(i)+']='+str(peak))

        if peak is not None:
            # print('adding peak= '+str(peak))

            if not lane_list[i].detected:
                temp_fit, temp_lane_inds, rectangles = sliding_window_polyfit_all(img_bin,peak)
                rectangle_img = create_image_of_sliding_windows_polyfit(rectangle_img,img_bin, temp_fit, temp_lane_inds, rectangles,colour=(255,255,0))
            else:
                temp_fit, temp_lane_inds=polyfit_using_prev_fit_all(img_bin,lane_list[i].best_fit)
                rectangle_img= create_image_of_polyfit_using_prev_fit(rectangle_img,img_bin,temp_fit,temp_lane_inds)


            ignored=lane_list[i].add_fit(temp_fit, temp_lane_inds,i,peak)


            print('ignored is: ' + str(ignored)+' ['+str(i)+']')
            print(str(four_lanes_before[i]))
            print(str(four_lanes_after[i]))
            print(str(four_lanes_before[i] is not four_lanes_after[i]))
            if ignored is True and four_lanes_before[i] is not four_lanes_after[i]:
                print('reverse_switch_lanes activated for ['+str(i)+']')
                reverse_switch_lanes(four_lanes_before)
                print('lane peaks after reverse_switch_lanes:'+str(lane_list[0].peak)+'///'+str(lane_list[1].peak)+'///'+str(lane_list[2].peak)+'///'+str(lane_list[3].peak))
                four_lanes_after=four_lanes_before


            # print('peak after adding= '+ str(lane_list[i].peak))
            # print('___________________________________________')
        else:
            # if lane_list[i].detected is not None:

                if lane_list[i].confidence < 10 and lane_list[i].best_fit is not None:
                    lane_list[i].reset_lane()
                    #print('lane_list['+ str(i)+'].reset_lane()')
                    #lane_list[i].detected=False
                elif lane_list[i].confidence-3 >= 0:
                    lane_list[i].confidence-= 3
                    #follow line function
                    lane_correction=follow_other_lane(i)
                    lane_list[i].best_fit[2]+=lane_correction
            # elif lane_list[i].confidence-5 >= 0:
            #     lane_list[i].confidence-= 5

            #lane_list[i].detected = False


    # for i, peak in enumerate(four_lanes):
    #     # print('current self.peak['+str(i)+']='+str(lane_list[i].peak))
    #
    #     if peak is not None:
    #         # print('adding peak= '+str(peak))
    #
    #
    #
    #         if not lane_list[i].detected:
    #             temp_fit, temp_lane_inds, rectangles = sliding_window_polyfit_all(img_bin,peak)
    #             rectangle_img = create_image_of_sliding_windows_polyfit(rectangle_img,img_bin, temp_fit, temp_lane_inds, rectangles,colour=(255,255,0))
    #         else:
    #             temp_fit, temp_lane_inds=polyfit_using_prev_fit_all(img_bin,lane_list[i].best_fit)
    #             rectangle_img= create_image_of_polyfit_using_prev_fit(rectangle_img,img_bin,temp_fit,temp_lane_inds)
    #
    #
    #         ignored=lane_list[i].add_fit(temp_fit, temp_lane_inds,i,peak)
    #
    #
    #         print('ignored is: ' + str(ignored)+' ['+str(i)+']')
    #         print(str(four_lanes_before[i]))
    #         print(str(four_lanes_after[i]))
    #         print(str(four_lanes_before[i] is not four_lanes_after[i]))
    #         if ignored is True and four_lanes_before[i] is not four_lanes_after[i]:
    #             print('reverse_switch_lanes activated for ['+str(i)+']')
    #             reverse_switch_lanes(four_lanes_before)
    #             print('lane peaks after reverse_switch_lanes:'+str(lane_list[0].peak)+'///'+str(lane_list[1].peak)+'///'+str(lane_list[2].peak)+'///'+str(lane_list[3].peak))
    #             four_lanes_after=four_lanes_before
    #
    #
    #         # print('peak after adding= '+ str(lane_list[i].peak))
    #         # print('___________________________________________')
    #     else:
    #         #if lane_list[i].peak is not None:
    #
    #         if lane_list[i].confidence < 10 and lane_list[i].best_fit is not None:
    #             lane_list[i].reset_lane()
    #             #print('lane_list['+ str(i)+'].reset_lane()')
    #             #lane_list[i].detected=False
    #         elif lane_list[i].confidence-3 >= 0:
    #             lane_list[i].confidence-= 3
    #             #follow line function
    #             lane_correction=follow_other_lane(i)
    #             lane_list[i].best_fit[2]+=lane_correction
    #         # elif lane_list[i].confidence-5 >= 0:
    #         #     lane_list[i].confidence-= 5
    #
    #         #lane_list[i].detected = False


    # lanelisttt=(lane_list[0].best_fit[2],lane_list[1].best_fit[2],lane_list[2].best_fit[2],lane_list[3].best_fit[2])

# otkomentirati
    invalid_lane_order=check_lane_order()
    if invalid_lane_order is True:
        lane_list[0].reset_lane()
        lane_list[1].reset_lane()
        lane_list[2].reset_lane()
        lane_list[3].reset_lane()



        # print('lane_list.bestfit[2]')
        # print(str(lane_list[0].best_fit[2]))
        # print(str(lane_list[1].best_fit[2]))
        # print(str(lane_list[2].best_fit[2]))
        # print(str(lane_list[3].best_fit[2]))
        # print('-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/')


    # print('///////////////////////////////////////////////////////')
    # print('after self.previous_fit[0]: '+str(lane_list[0].previous_fit))
    # print('after self.previous_fit[1]: '+str(lane_list[1].previous_fit))
    # print('after self.previous_fit[2]: '+str(lane_list[2].previous_fit))
    # print('after self.previous_fit[3]: '+str(lane_list[3].previous_fit))
    # print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

    # print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    # print('********************************************')
    # print('peaks after adding: self.peaks=['+str(lane_list[0].peak)+' '+str(lane_list[1].peak)+' '+str(lane_list[2].peak)+' '+str(lane_list[3].peak)+']')
    # print('********************************************')
    return rectangle_img

# def update_lanes(img_bin,rectangle_img,four_lanes):
#     for i, peak in enumerate(four_lanes):
#         if peak is not None:
#
#             if not lane_list[i].detected:
#
#
#             #l_fit, l_lane_inds, rectangles = sliding_window_polyfit_all(img_bin,peak)
#                 temp_fit, temp_lane_inds, rectangles = sliding_window_polyfit_all(img_bin,peak)
#
#             #print(l_fit)
#             #print(l_fit.shape)
#             #if l_fit[0] < 0 :
#             #condition for vertical curves
#                 if temp_fit[2] >= 0 and -1 <= temp_fit[1] <= 1:
#                     #print('added')
#                     #print(' ')
#
#                     lane_list[i].add_fit(temp_fit, temp_lane_inds,i,peak)
#
#                     #print(temp_fit, temp_lane_inds)
#                     #print('')
#                     #lista.append(temp_fit)
#                     rectangle_img = create_image_of_sliding_windows_polyfit(rectangle_img,img_bin, temp_fit, temp_lane_inds, rectangles,colour=(255,255,0))
#             else:
#                 temp_fit, temp_lane_inds=polyfit_using_prev_fit_all(img_bin,lane_list[i].best_fit)
#                 lane_list[i].add_fit(temp_fit, temp_lane_inds,i,peak)
#
#
#                 rectangle_img= create_image_of_polyfit_using_prev_fit(rectangle_img,img_bin,temp_fit,temp_lane_inds)
#
#         else:
#             lane_list[i].reset_lane()
#             #lane_list[i].detected = False
#     return rectangle_img


def process_image_4lanes(imgOriginal,fullscreen=False):

    #processing image and returning binary image
    h,w = imgOriginal.shape[:2]
    new_img = np.copy(imgOriginal)

    if w==1280:
        font_size=1
        thickness=2
        position1=40
        position2=160
        inc=80
    elif w==640:
        font_size=0.5
        thickness=1
        position1=20
        position2=140
        inc=40
    else:
        font_size=1
        thickness=2
        position1=40
        position2=160
        inc=80
    #img_bin, Minv, img_unwarped = Lff.pipeline(new_img, diagnostic_images=True)
    img_bin, Minv, img_unwarped = IPF.pipeline(new_img)
    #peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)

    #peaks,histogram_image=find_4_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)

    peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)
    #peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)
    print('___________________________________')
    print('calculated peaks')
    print(peaks)

    cv2.line(histogram_image,(int(w*0.25),0),(int(w*0.25),h),(255,255,0),2)
    cv2.line(histogram_image,(int(w*0.5),0),(int(w*0.5),h),(255,255,0),2)
    cv2.line(histogram_image,(int(w*0.75),0),(int(w*0.75),h),(255,255,0),2)
    cv2.line(histogram_image,(w,0),(w,h),(255,255,0),2)

    # peaks,histogram_image=find_44_histogram_peaks(peaks,histogram_image,(np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)))
    # print(peaks)
    # print('')
#--------------------------------------------------------------------------------
    rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)

    four_lanes, lane_change = allocate_peaks_to_4lanes(peaks)

    print('sorted peaks')
    print(four_lanes)
    print('___________________________________')

    cv2.putText(histogram_image, 'Calculated peaks: '+str(peaks), (40,80), cv2.FONT_HERSHEY_DUPLEX, float(w/1028), (0,0,255), int(w/640), cv2.LINE_AA)
    cv2.putText(histogram_image, 'Sorted peaks: '+str(four_lanes), (40,115), cv2.FONT_HERSHEY_DUPLEX, float(w/1028), (0,0,255), int(w/640), cv2.LINE_AA)

    rectangle_img=update_lanes(img_bin,rectangle_img,four_lanes)

    img_out1=np.copy(imgOriginal)
    img_out2=np.copy(img_bin)
    img_out2= np.dstack((img_bin*255, img_bin*255, img_bin*255))

    print('LINIJE')
    print(lane_list[0].best_fit)
    print(lane_list[1].best_fit)
    print(lane_list[2].best_fit)
    print(lane_list[3].best_fit)
    print('LINIJE')

    # natpis=False
    # correctionIndex=0

    natpis,correctionIndex=correct_false_lines_abcoef()
    if natpis is True:
        cv2.putText(img_out2,'B_corrected'+str(correctionIndex+1)+': '+str(lane_list[correctionIndex].best_fit[0])+' '+str(lane_list[correctionIndex].best_fit[1])+' '+str(lane_list[correctionIndex].best_fit[2])+' ', (100,700), cv2.FONT_HERSHEY_DUPLEX, font_size, (255,0,200), thickness, cv2.LINE_AA)
        # #cv2.putText(img_out2,'A_corrected'+str(correctionIndex)+': '+str(lane_list[correctionIndex].best_fit[0])+' '+str(lane_list[correctionIndex].best_fit[1])+' '+str(lane_list[correctionIndex].best_fit[2])+' ', (100,620), cv2.FONT_HERSHEY_DUPLEX, font_size, (255,0,200), thickness, cv2.LINE_AA)



    #print('_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*')


    for i, elements in enumerate(lane_list):
        if lane_list[i].best_fit is not None:
            img_out1 = draw_lane_custom(img_out1, img_bin, lane_list[i].best_fit, Minv)
            img_out2 = draw_all_curves_custom(img_out2, lane_list[i])
    for i, elements in enumerate(lane_list):
        if lane_list[i].best_fit is not None:
            cv2.putText(img_out2,str(i+1)+'. '+ str(lane_list[i].best_fit[0])+' '+str(lane_list[i].best_fit[1])+' '+str(lane_list[i].best_fit[2])+' ', (position1,position2+i*inc), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)


    # print(str(lane_list[0].best_fit))
    # print('__________________________')
    # print(str(lane_list[1].best_fit))
    # print('__________________________')
    # print(str(lane_list[2].best_fit))
    # print('__________________________')
    # print(str(lane_list[3].best_fit))
    # print('__________________________')
    # print('__________________________')
    img_out1=highlight_road(img_out1, img_bin, [lane_list[0].best_fit,lane_list[1].best_fit,lane_list[2].best_fit,lane_list[3].best_fit], Minv)
    #
    cv2.putText(img_out1, 'Confidence:', (800,40), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(img_out1, 'Line0  Line1  Line2  Line3', (800,80), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(img_out1, str(lane_list[0].confidence)+'%    '+str(lane_list[1].confidence)+'%    '+str(lane_list[2].confidence)+'%   '+str(lane_list[3].confidence)+'%', (800,120), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # cv2.putText(img_out1, 'Line1 confidence: '+str(lane_list[1].confidence)+'%', (880,80), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    # cv2.putText(img_out1, 'Line2 confidence: '+str(lane_list[2].confidence)+'%', (880,120), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    # cv2.putText(img_out1, 'Line3 confidence: '+str(lane_list[3].confidence)+'%', (880,160), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)


#-------------------------------------------------------------------------------------
    #processed_image=np.copy(imgOriginal)
    if fullscreen is False:
    #output image is diagnostic
        final_image = combine_images_smaller(img_out1,img_unwarped,img_bin,histogram_image,rectangle_img,img_out2)
    else:
    #output image is result only
        final_image=img_out1

    return final_image

def process_image_for_validation(imgOriginal,save_folder_path,compare_image_path,fullscreen=False):

    #processing image and returning binary image
    h,w = imgOriginal.shape[:2]
    new_img = np.copy(imgOriginal)

    if w==1280:
        font_size=1
        thickness=2
        position1=40
        position2=160
        inc=80
    elif w==640:
        font_size=0.5
        thickness=1
        position1=20
        position2=140
        inc=40
    else:
        font_size=1
        thickness=2
        position1=40
        position2=160
        inc=80
    #img_bin, Minv, img_unwarped = Lff.pipeline(new_img, diagnostic_images=True)
    img_bin, Minv, img_unwarped = IPF.pipeline(new_img)
    #peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)

    #peaks,histogram_image=find_4_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)

    peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)
    #peaks,histogram_image=find_histogram_peaks((np.sum(img_bin[:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)
    # print('___________________________________')
    # print('calculated peaks')
    # print(peaks)

    cv2.line(histogram_image,(int(w*0.25),0),(int(w*0.25),h),(255,255,0),2)
    cv2.line(histogram_image,(int(w*0.5),0),(int(w*0.5),h),(255,255,0),2)
    cv2.line(histogram_image,(int(w*0.75),0),(int(w*0.75),h),(255,255,0),2)
    cv2.line(histogram_image,(w,0),(w,h),(255,255,0),2)

    # peaks,histogram_image=find_44_histogram_peaks(peaks,histogram_image,(np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)))
    # print(peaks)
    # print('')
#--------------------------------------------------------------------------------
    rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)

    four_lanes, lane_change = allocate_peaks_to_4lanes(peaks)

    # print('sorted peaks')
    # print(four_lanes)
    # print('___________________________________')

    cv2.putText(histogram_image, 'Calculated peaks: '+str(peaks), (40,80), cv2.FONT_HERSHEY_DUPLEX, float(w/1028), (0,0,255), int(w/640), cv2.LINE_AA)
    cv2.putText(histogram_image, 'Sorted peaks: '+str(four_lanes), (40,115), cv2.FONT_HERSHEY_DUPLEX, float(w/1028), (0,0,255), int(w/640), cv2.LINE_AA)

    rectangle_img=update_lanes(img_bin,rectangle_img,four_lanes)

    img_out1=np.copy(imgOriginal)
    img_out2=np.copy(img_bin)
    img_out2= np.dstack((img_bin*255, img_bin*255, img_bin*255))

    # print('LINIJE')
    # print(lane_list[0].best_fit)
    # print(lane_list[1].best_fit)
    # print(lane_list[2].best_fit)
    # print(lane_list[3].best_fit)
    # print('LINIJE')

    # natpis=False
    # correctionIndex=0

    natpis,correctionIndex=correct_false_lines_abcoef()
    if natpis is True:
        cv2.putText(img_out2,'B_corrected'+str(correctionIndex+1)+': '+str(lane_list[correctionIndex].best_fit[0])+' '+str(lane_list[correctionIndex].best_fit[1])+' '+str(lane_list[correctionIndex].best_fit[2])+' ', (100,700), cv2.FONT_HERSHEY_DUPLEX, font_size, (255,0,200), thickness, cv2.LINE_AA)
        # #cv2.putText(img_out2,'A_corrected'+str(correctionIndex)+': '+str(lane_list[correctionIndex].best_fit[0])+' '+str(lane_list[correctionIndex].best_fit[1])+' '+str(lane_list[correctionIndex].best_fit[2])+' ', (100,620), cv2.FONT_HERSHEY_DUPLEX, font_size, (255,0,200), thickness, cv2.LINE_AA)



    #print('_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*')


    for i, elements in enumerate(lane_list):
        if lane_list[i].best_fit is not None:
            img_out1 = draw_lane_custom(img_out1, img_bin, lane_list[i].best_fit, Minv)
            img_out2 = draw_all_curves_custom(img_out2, lane_list[i])
    for i, elements in enumerate(lane_list):
        if lane_list[i].best_fit is not None:
            cv2.putText(img_out2,str(i+1)+'. '+ str(lane_list[i].best_fit[0])+' '+str(lane_list[i].best_fit[1])+' '+str(lane_list[i].best_fit[2])+' ', (position1,position2+i*inc), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)

    pic_for_validation,combined_pic=create_validate_image(img_bin, (lane_list[0].best_fit,lane_list[1].best_fit,lane_list[2].best_fit,lane_list[3].best_fit), Minv, compare_image_path)
    # new_save_folder_path = save_folder_path.replace(".jpg", ".png")
    cv2.imwrite(save_folder_path,pic_for_validation)
    cv2.imwrite(save_folder_path+'_combined',combined_pic)
    # print(str(lane_list[0].best_fit))
    # print('__________________________')
    # print(str(lane_list[1].best_fit))
    # print('__________________________')
    # print(str(lane_list[2].best_fit))
    # print('__________________________')
    # print(str(lane_list[3].best_fit))
    # print('__________________________')
    # print('__________________________')
    img_out1=highlight_road(img_out1, img_bin, [lane_list[0].best_fit,lane_list[1].best_fit,lane_list[2].best_fit,lane_list[3].best_fit], Minv)
    #
    cv2.putText(img_out1, 'Confidence:', (800,40), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(img_out1, 'Line0  Line1  Line2  Line3', (800,80), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(img_out1, str(lane_list[0].confidence)+'%    '+str(lane_list[1].confidence)+'%    '+str(lane_list[2].confidence)+'%   '+str(lane_list[3].confidence)+'%', (800,120), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # cv2.putText(img_out1, 'Line1 confidence: '+str(lane_list[1].confidence)+'%', (880,80), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    # cv2.putText(img_out1, 'Line2 confidence: '+str(lane_list[2].confidence)+'%', (880,120), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    # cv2.putText(img_out1, 'Line3 confidence: '+str(lane_list[3].confidence)+'%', (880,160), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    counter=[0,0,0,0]
    for i,lane in enumerate(lane_list):
        if lane_list[i].best_fit is not None:
            counter[i]=1
    print('number of lane lanes: '+str(counter))
#-------------------------------------------------------------------------------------
    #processed_image=np.copy(imgOriginal)


    return pic_for_validation,counter

def draw_all_curves(img_bin, l_line, r_line):
    img_bin_fit = np.copy(img_bin)
    img_bin_fit = np.dstack((img_bin*255, img_bin*255, img_bin*255))

    for i, fit in enumerate(l_line.current_fit):
        img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (20*i+100,0,20*i+100))
    for i, fit in enumerate(r_line.current_fit):
        img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (0,20*i+100,20*i+100))
    cv2.putText(img_bin_fit, "6. Overhead with all fits added", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    img_bin_fit = plot_fit_onto_img(img_bin_fit, l_line.best_fit, (255,255,0))
    img_bin_fit = plot_fit_onto_img(img_bin_fit, r_line.best_fit, (255,255,0))
    #diag_img[int(height/3)*2:height,int(width/3)*2:width-2,:] = cv2.resize(img_bin_fit,(int(width/3),int(height/3)))

    return img_bin_fit

def draw_all_curves_custom(img_bin_fit, l_line):
    #img_bin_fit = np.copy(img_bin)
    #img_bin_fit = np.dstack((img_bin*255, img_bin*255, img_bin*255))
    for i, fit in enumerate(l_line.current_fit):
        img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (20*i+100,0,20*i+100))

    h,w=img_bin_fit.shape[:2]
    if w==1280:
        font_size=2
        thickness=2
        position=(40,80)
    elif w==640:
        font_size=1
        thickness=1
        position=(20,40)
    else:
        font_size=2
        thickness=2
        position=(40,80)
    cv2.putText(img_bin_fit, "6. Overhead with all fits added", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,255,0), thickness, cv2.LINE_AA)
    img_bin_fit = plot_fit_onto_img(img_bin_fit, l_line.best_fit, (255,255,0))
    #diag_img[int(height/3)*2:height,int(width/3)*2:width-2,:] = cv2.resize(img_bin_fit,(int(width/3),int(height/3)))
    return img_bin_fit


def create_image_of_sliding_windows_polyfit(img_bin,l_fit,r_fit,l_lane_inds,r_lane_inds, rectangles):
    rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
    rectangle_img = plot_fit_onto_img(rectangle_img,l_fit,(0,255,255))
    rectangle_img = plot_fit_onto_img(rectangle_img,r_fit,(0,255,255))

    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    rectangle_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
    rectangle_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]

    h,w=img_bin.shape
    if w==1280:
        font_size=2
        thickness=2
        position=(40,80)
    elif w==640:
        font_size=1
        thickness=1
        position=(20,40)
    else:
        font_size=2
        thickness=2
        position=(40,80)

    cv2.putText(rectangle_img, "5. sliding_window_polyfit", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)

    for rect in rectangles:
        # Draw the windows on the visualization image
        cv2.rectangle(rectangle_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)
        cv2.rectangle(rectangle_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2)
    return rectangle_img


def create_image_of_polyfit_using_prev_fit(img_bin, l_fit, r_fit, l_lane_inds, r_lane_inds):
    rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
    rectangle_img = plot_fit_onto_img(rectangle_img,l_fit,(0,255,255))
    rectangle_img = plot_fit_onto_img(rectangle_img,r_fit,(0,255,255))

    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    rectangle_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
    rectangle_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]

    cv2.putText(rectangle_img, "5. polyfit_using_prev_fit", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

    return rectangle_img


def create_image_of_polyfit_using_prev_fit(rectangle_img, img_bin, l_fit, l_lane_inds):
    #rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
    rectangle_img = plot_fit_onto_img(rectangle_img,l_fit,(0,255,255))


    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    h,w=img_bin.shape
    if w==1280:
        font_size=2
        thickness=2
        position=(40,80)
    elif w==640:
        font_size=1
        thickness=1
        position=(20,40)
    else:
        font_size=2
        thickness=2
        position=(40,80)

    rectangle_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]

    cv2.putText(rectangle_img, "5. polyfit_using_prev_fit", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,255,0), thickness, cv2.LINE_AA)

    return rectangle_img


def sliding_window_polyfit_all(img,peak):

    if peak is None:
        return None, None, None
    #for using middle quarters
    leftx_base = peak

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
    # Set the width of the windows +/- margin
    #margin = 100
    #margin = 60
    margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)


    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    left_fit  = None
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)

    return left_fit, left_lane_inds, rectangle_data

def create_image_of_sliding_windows_polyfit(rectangle_img,img_bin,fit,lane_inds, rectangles,colour=(0,0,255)):
    #rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
    rectangle_img =plot_fit_onto_img(rectangle_img,fit,(0,255,255))
    #rectangle_img = plot_fit_onto_img(rectangle_img,r_fit,(0,255,255))

    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    rectangle_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [colour[0], colour[1], colour[2]]

    h,w=img_bin.shape
    if w==1280:
        font_size=2
        thickness=2
        position=(40,80)
    elif w==640:
        font_size=1
        thickness=1
        position=(20,40)
    else:
        font_size=2
        thickness=2
        position=(40,80)
    #rectangle_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]
    cv2.putText(rectangle_img, "5. sliding_window_polyfit", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)

    for rect in rectangles:
        # Draw the windows on the visualization image
        cv2.rectangle(rectangle_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)
        #cv2.rectangle(rectangle_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2)
    return rectangle_img

def combine_images(img_original,img_unwarp,img_bin,histogram_image,curves_images,img_bin_fit,processed_img):
    combined_image = np.zeros((960,1280,3), dtype=np.uint8)
    height,width,_=combined_image.shape

#---------------------------------------------------------------------------------------
    # original output (top left)
    cv2.putText(img_original, "1. Original image ->", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    smaller_img_out2=  cv2.resize(img_original,(int(width/3),int(height/4)))
    combined_image[0:int(height/4),0:int(width/3)] =smaller_img_out2

#---------------------------------------------------------------------------------------

    # warped imapge (top middle)
    cv2.putText(img_unwarp, "2. Warped image ->", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    smaller_warped_img=  cv2.resize(img_unwarp,(int(width/3),int(height/4)))
    combined_image[0:int(height/4),int(width/3):2*int(width/3)] =smaller_warped_img
#---------------------------------------------------------------------------------------

    # binary overhead view (top right)
    img_bin2=np.copy(img_bin)
    img_bin2 = np.dstack((img_bin*255, img_bin*255, img_bin*255))
    cv2.putText(img_bin2, "3. Binary image v", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    resized_img_bin = cv2.resize(img_bin2,(int(width/3),int(height/4)))
    r_height, r_width, _ = resized_img_bin.shape
    cv2.line(resized_img_bin,(0,r_height//2),(r_width,r_height//2),(0,0,255),1)
    combined_image[0:int(height/4),2*int(width/3):width-2] = resized_img_bin
#--------------------------------------------------------------------------------------------------------
    #histogram image (middle right)
    # histogram = np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)
    # histogram_image=np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)
    # #histogram_image=np.ones((binary_image.shape[0]//2,binary_image.shape[1]),dtype=int)
    # out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)
    # i=1
    # while i <= len(histogram)-1:
    #     #histogram_image[histogram_image.shape[0]-int(histogram[i]),i]=0
    #     cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(255,255,255),2)
    #     i+=1
    # cv2.putText(out_image, "4. Histogram image", (40,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
    combined_image[int(height/4):int(height/4)*2,int(width/3)*2:int(width/3)*3,:]=cv2.resize(histogram_image,(int(width/3),int(height/4)))

#---------------------------------------------------------------------------------------
    #image of curves (middle middle)
    #cv2.putText(curves_images, "<- lane find", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    smaller_curves_images=  cv2.resize(curves_images,(int(width/3),int(height/4)))
    combined_image[int(height/4):int(height/4)*2,int(width/3):2*int(width/3)] =smaller_curves_images

#------------------------------------------------------------------------------------------
    # overhead with all fits added (middle left)
    combined_image[int(height/4):int(height/4)*2,0:int(width/3),:] = cv2.resize(img_bin_fit,(int(width/3),int(height/4)))
#------------------------------------------------------------------------------------------------------

    # original processed output
    processed_image=  cv2.resize(processed_img,(width,int(height/2)))
    combined_image[int(height/2):height,0:width] =processed_image


#----------------------------------------------------------------------------------------------------------
    return combined_image


def combine_images_smaller(img_original,img_unwarp,img_bin,histogram_image,curves_images,all_curves_image):
    combined_image = np.zeros((960,1280,3), dtype=np.uint8)
    height,width,_=combined_image.shape

    h,w=img_original.shape[:2]
    if w==1280:
        font_size=2
        thickness=2
        position=(40,80)
        position2=(40,140)
    elif w==640:
        font_size=1
        thickness=1
        position=(20,40)
        position2=(20,80)
    else:
        font_size=2
        thickness=2
        position=(40,80)
        position2=(40,140)
#---------------------------------------------------------------------------------------
    # original output (top left)
    cv2.putText(img_original, "1. Original image", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,255,0), thickness, cv2.LINE_AA)
    combined_image[0:int(height/3),0:int(width/2)] =cv2.resize(img_original,(int(width/2),int(height/3)))

#---------------------------------------------------------------------------------------

    # warped imapge (top middle)
    cv2.putText(img_unwarp, "2. Warped image", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,255,0), thickness, cv2.LINE_AA)
    combined_image[0:int(height/3),int(width/2):int(width/2)*2] = cv2.resize(img_unwarp,(int(width/2),int(height/3)))

#---------------------------------------------------------------------------------------

    # binary overhead view (top right)
    img_bin2=np.copy(img_bin)
    img_bin2 = np.dstack((img_bin*255, img_bin*255, img_bin*255))
    cv2.putText(img_bin2, "3. Binary image", position, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,255,0), thickness, cv2.LINE_AA)

    if len(FP.binary_combinations) is 1:
        cv2.putText(img_bin2, str(FP.binary_combinations[0]), position2, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
    if len(FP.binary_combinations) is 2:
        cv2.putText(img_bin2, str(FP.binary_combinations[0])+'+'+str(FP.binary_combinations[1]), position2, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
    if len(FP.binary_combinations) is 3:
        cv2.putText(img_bin2, str(FP.binary_combinations[0])+'+'+str(FP.binary_combinations[1])+'+'+str(FP.binary_combinations[2]), position2, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
    if len(FP.binary_combinations) is 4:
        cv2.putText(img_bin2, str(FP.binary_combinations[0])+'+'+str(FP.binary_combinations[1])+'+'+str(FP.binary_combinations[2])+'+'+str(FP.binary_combinations[3]), position2, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
    if len(FP.binary_combinations) is 5:
        cv2.putText(img_bin2, str(FP.binary_combinations[0])+'+'+str(FP.binary_combinations[1])+'+'+str(FP.binary_combinations[2])+'+'+str(FP.binary_combinations[3])+'+'+str(FP.binary_combinations[4]), position2, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
    if len(FP.binary_combinations) is 6:
        cv2.putText(img_bin2, str(FP.binary_combinations[0])+'+'+str(FP.binary_combinations[1])+'+'+str(FP.binary_combinations[2])+'+'+str(FP.binary_combinations[3])+'+'+str(FP.binary_combinations[4])+'+'+str(FP.binary_combinations[5]), position2, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
    if len(FP.binary_combinations) is 7:
        cv2.putText(img_bin2, str(FP.binary_combinations[0])+'+'+str(FP.binary_combinations[1])+'+'+str(FP.binary_combinations[2])+'+'+str(FP.binary_combinations[3])+'+'+str(FP.binary_combinations[4])+'+'+str(FP.binary_combinations[5])+'+'+str(FP.binary_combinations[6]), position2, cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)


    resized_img_bin = cv2.resize(img_bin2,(int(width/2),int(height/3)))
    r_height, r_width, _ = resized_img_bin.shape
    cv2.line(resized_img_bin,(0,r_height//2),(r_width,r_height//2),(0,0,255),1)
    combined_image[int(height/3):int(height/3)*2,0:int(width/2)] = resized_img_bin
#--------------------------------------------------------------------------------------------------------
    #histogram image (middle right)
    combined_image[int(height/3)*2:int(height/3)*3,0:int(width/2)]=cv2.resize(histogram_image,(int(width/2),int(height/3)))

#---------------------------------------------------------------------------------------
    #image of curves (middle middle)
    #cv2.putText(curves_images, "<- lane find", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
    combined_image[int(height/3):int(height/3)*2,int(width/2):int(width)] =cv2.resize(curves_images,(int(width/2),int(height/3)))

#------------------------------------------------------------------------------------------
    combined_image[int(height/3)*2:int(height),int(width/2):int(width)] =cv2.resize(all_curves_image,(int(width/2),int(height/3)))
#------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------
    return combined_image


def dynamic_calibration_information(original_img):
    height,width,_=original_img.shape
    image = np.zeros((300, 300, 3), np.uint8)
    image[:] = (0, 0, 255)

    result=np.copy(original_img)

    result[0:int(height/3),0:int(width/2)] =cv2.resize(img_original,(int(width/2),int(height/3)))



    return result
