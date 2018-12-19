import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Image_processing_functions as IPF


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
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                # bad fit! abort! abort! ... well, unless there are no fits in the current_fit queue, then we'll take it
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
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

l_line = Line()
r_line = Line()

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

def find_histogram_peaks(histogram,histogram_image, image=False):

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
#
# def process_image(img, diagnostic_output=False):
#     #copy input image
#     new_img = np.copy(img)
#     #get binary image
#     if diagnostic_output is True:
#         img_bin, Minv, img_unwarp = pipeline(new_img, diagnostic_output)
#     else:
#         img_bin, Minv = pipeline(new_img, diagnostic_output)
#
#     height,width,_ = new_img.shape
#
#
#     # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
#     if not l_line.detected or not r_line.detected:
#         l_fit, r_fit, l_lane_inds, r_lane_inds, rectangles = sliding_window_polyfit(img_bin)
#         fit_method=2
#     else:
#         l_fit, r_fit, l_lane_inds, r_lane_inds = polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)
#         fit_method=1
#
#
#     # invalidate both fits if the difference in their x-intercepts isn't around 350 px (+/- 100 px)
#     if l_fit is not None and r_fit is not None:
#         # calculate x-intercept (bottom of image, x=image_height) for fits
#         h = img.shape[0]
#         l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
#         r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
#         x_int_diff = abs(r_fit_x_int-l_fit_x_int)
#         if abs(350 - x_int_diff) > 100:
#             l_fit = None
#             r_fit = None
#
#     l_line.add_fit(l_fit, l_lane_inds)
#     r_line.add_fit(r_fit, r_lane_inds)
#
#     # draw the current best fit if it exists
#     if l_line.best_fit is not None and r_line.best_fit is not None:
#         img_out1 = draw_lane(new_img, img_bin, l_line.best_fit, r_line.best_fit, Minv)
#         rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
#                                                                l_lane_inds, r_lane_inds)
#         img_out = draw_data(img_out1, (rad_l+rad_r)/2, d_center)
#     else:
#         img_out = new_img
#
# #---------------------------------------------------------------------------------------
#     #diagnostic_output = True
#     if diagnostic_output:
#
#
#         # put together multi-view output
#         diag_img = np.zeros((720,1280,3), dtype=np.uint8)
# #---------------------------------------------------------------------------------------
# #lane finding method (middle right)
#         rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
#         rectangle_img = plot_fit_onto_img(rectangle_img,l_fit,(0,255,255))
#         rectangle_img = plot_fit_onto_img(rectangle_img,r_fit,(0,255,255))
#
#         nonzero = img_bin.nonzero()
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#
#         rectangle_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
#         rectangle_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]
#
#         if fit_method==1:
#             cv2.putText(rectangle_img, "polyfit_using_prev_fit", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
#         elif fit_method==2:
#             cv2.putText(rectangle_img, "sliding_window_polyfit", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
#
#             for rect in rectangles:
#                 # Draw the windows on the visualization image
#                 cv2.rectangle(rectangle_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)
#                 cv2.rectangle(rectangle_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2)
#
#
#         smaller_window_img=  cv2.resize(rectangle_img,(int(width/3),int(height/3)))
#         diag_img[int(height/3):int(height/3)*2,int(width/3)*2:width-2] =smaller_window_img
# #------------------------------------------------------------------------------------------
#
#         # original processed output
#         smaller_img_out=  cv2.resize(img_out,(int(width/3)*2,int(height/3)*2))
#         diag_img[int(height/3):height,0:int(width/3)*2] =smaller_img_out
#
# #---------------------------------------------------------------------------------------
#
#         # original output (top left)
#         cv2.putText(img, "1. Original image", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
#         smaller_img_out2=  cv2.resize(img,(int(width/3),int(height/3)))
#         diag_img[0:int(height/3),0:int(width/3)] =smaller_img_out2
#
# #---------------------------------------------------------------------------------------
#
#         # warped imapge (top middle)
#         cv2.putText(img_unwarp, "2. Warped image", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
#         smaller_warped_img=  cv2.resize(img_unwarp,(int(width/3),int(height/3)))
#         diag_img[0:int(height/3),int(width/3):2*int(width/3)] =smaller_warped_img
# #---------------------------------------------------------------------------------------
#
#         # binary overhead view (top right)
#         img_bin2=np.copy(img_bin)
#         img_bin2 = np.dstack((img_bin*255, img_bin*255, img_bin*255))
#         cv2.putText(img_bin2, "3. Binary image", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
#         resized_img_bin = cv2.resize(img_bin2,(int(width/3),int(height/3)))
#         r_height, r_width, _ = resized_img_bin.shape
#         cv2.line(resized_img_bin,(0,r_height//2),(r_width,r_height//2),(0,0,255),1)
#         diag_img[0:int(height/3),2*int(width/3):width-2] = resized_img_bin
#
# #---------------------------------------------------------------------------------------
#
#         # overhead with all fits added (bottom right)
#         img_bin_fit = np.copy(img_bin)
#         img_bin_fit = np.dstack((img_bin*255, img_bin*255, img_bin*255))
#         for i, fit in enumerate(l_line.current_fit):
#             img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (20*i+100,0,20*i+100))
#         for i, fit in enumerate(r_line.current_fit):
#             img_bin_fit = plot_fit_onto_img(img_bin_fit, fit, (0,20*i+100,20*i+100))
#         cv2.putText(img_bin_fit, "Overhead with all fits added", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
#         img_bin_fit = plot_fit_onto_img(img_bin_fit, l_line.best_fit, (255,255,0))
#         img_bin_fit = plot_fit_onto_img(img_bin_fit, r_line.best_fit, (255,255,0))
#         diag_img[int(height/3)*2:height,int(width/3)*2:width-2,:] = cv2.resize(img_bin_fit,(int(width/3),int(height/3)))
# #------------------------------------------------------------------------------------------------------
#
#         img_out = diag_img
#
#     return img_out

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


def create_image_of_sliding_windows_polyfit(img_bin,l_fit,r_fit,l_lane_inds,r_lane_inds, rectangles):
    rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
    rectangle_img = plot_fit_onto_img(rectangle_img,l_fit,(0,255,255))
    rectangle_img = plot_fit_onto_img(rectangle_img,r_fit,(0,255,255))

    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    rectangle_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
    rectangle_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]



    cv2.putText(rectangle_img, "5. sliding_window_polyfit", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

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
