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

def main():
    dashcam_image_path = '/home/profesor/Documents/[ADAS]_Finding_Lanes/dashcam_driving/'
    img_arg="frame"
    count = 500
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
#----------------------------------------------------------------------------------
        #make picture with histogram
        #histogram_image=Lff.draw_histogram(img_bin)

        #or make picture with histogram and peaks
        #peaks,histogram_image=Lff.find_histogram_peaks(histogram,histogram_image2,image=True)
        peaks,histogram_image=Lff.find_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)
#--------------------------------------------------------------------------------
        #processing binary image to find lanes as curves
        height,width,_ = new_img.shape

        # if both left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
        if not l_line.detected or not r_line.detected:
            l_fit, r_fit, l_lane_inds, r_lane_inds, rectangles = Lff.sliding_window_polyfit(img_bin)
            curves_image = Lff.create_image_of_sliding_windows_polyfit(img_bin, l_fit, r_fit, l_lane_inds, r_lane_inds, rectangles)
        else:
            l_fit, r_fit, l_lane_inds, r_lane_inds = Lff.polyfit_using_prev_fit(img_bin, l_line.best_fit, r_line.best_fit)
            curves_image = Lff.create_image_of_polyfit_using_prev_fit(img_bin, l_fit, r_fit, l_lane_inds, r_lane_inds)


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
        img_all_fits = Lff.draw_all_curves(img_bin, l_line, r_line)
#---------------------------------------------------------------------------------------------------------
        # draw the current best fit if it exists
        if l_line.best_fit is not None and r_line.best_fit is not None:
            img_out1 = Lff.draw_lane(imgOriginal, img_bin, l_line.best_fit, r_line.best_fit, Minv)
            rad_l, rad_r, d_center = Lff.calc_curv_rad_and_center_dist(img_bin, l_line.best_fit, r_line.best_fit,
                                                                   l_lane_inds, r_lane_inds)
            processed_image = Lff.draw_data(img_out1, (rad_l+rad_r)/2, d_center)
        else:
            processed_image = new_img

#-------------------------------------------------------------------------------------
        #processed_image=np.copy(imgOriginal)
        final_image = Lff.combine_images(imgOriginal,img_unwarped,img_bin,histogram_image,curves_image,img_all_fits,processed_image)
#----------------------------------------------------------------------------------------------------------------------
        #final_image = Lff.process_image(imgOriginal)

#--------------------------------------------------------------------------------------------------------------------
        cv2.imshow(img_arg+str(count)+".jpg", final_image)
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
