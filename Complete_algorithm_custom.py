import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import Image_processing_functions as IPF

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


def sliding_window_polyfit_all(img,peak):

    #for using middle quarters
    leftx_base = peak

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
    # Set the width of the windows +/- margin
    margin = 100
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
    rectangle_img = Lff.plot_fit_onto_img(rectangle_img,fit,(0,255,255))
    #rectangle_img = plot_fit_onto_img(rectangle_img,r_fit,(0,255,255))

    nonzero = img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    rectangle_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [colour[0], colour[1], colour[2]]
    #rectangle_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]
    cv2.putText(rectangle_img, "5. sliding_window_polyfit", (40,80), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

    for rect in rectangles:
        # Draw the windows on the visualization image
        cv2.rectangle(rectangle_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)
        #cv2.rectangle(rectangle_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2)
    return rectangle_img

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
        #processing binary image to find lanes as curves
        #height,width,_ = new_img.shape

        peaks,histogram_image=Lff.find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)
        rectangle_img = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)

        for peak in peaks:

            l_fit, l_lane_inds, rectangles = sliding_window_polyfit_all(img_bin,peak)
            rectangle_img = create_image_of_sliding_windows_polyfit(rectangle_img,img_bin, l_fit, l_lane_inds, rectangles,colour=(255,255,0))

#----------------------------------------------------------------------------------------
        #histogram image (middle right)
        height,width=960,1280

        #choose between two types of histograms
        #histogram_image=Lff.draw_histogram(img_bin)

        img_all_fits=cv2.resize(histogram_image,(int(width/3),int(height/4)))
#--------------------------------------------------------------------------------------------------------
        #draw new image with all find lanes as curves and best line that represents curves
        #img_all_fits = Lff.draw_all_curves(img_bin, l_line, r_line)
#---------------------------------------------------------------------------------------------------------
        processed_image = new_img

#-------------------------------------------------------------------------------------
        #processed_image=np.copy(imgOriginal)
        #final_image = Lff.combine_images(imgOriginal,img_unwarped,img_bin,histogram_image,curves_image,img_all_fits,processed_image)
#----------------------------------------------------------------------------------------------------------------------
        #final_image = Lff.process_image(imgOriginal)

#--------------------------------------------------------------------------------------------------------------------
        cv2.imshow(img_arg+str(count)+".jpg", rectangle_img)
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
