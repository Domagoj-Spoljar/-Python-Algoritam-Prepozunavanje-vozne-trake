import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff
import Image_processing_functions as IPF
from scipy import signal
import function_parameters as FP

def main():
    dashcam_image_path = FP.dashcam_image_path
    count = FP.frame
    #dashcam_image_path = './Test_images/dashcam_driving/'
    img_arg="frame"
    #count = 705
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
        #peaks,histogram_image=Lff.find_4_histogram_peaks((np.sum(img_bin[img_bin.shape[0]//2:,:], axis=0)),(np.zeros((img_bin.shape[0]//2,img_bin.shape[1]),dtype=int)),image=True)
        cv2.putText(histogram_image, str(peaks), (40,100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
        combined_image = np.zeros((960,1280,3), dtype=np.uint8)
        height,width,_=combined_image.shape
        smaller_img_out2=  cv2.resize(histogram_image,(int(width),int(height/2)))
        img_bin = np.uint8(np.dstack((img_bin, img_bin, img_bin))*255)
        smaller_img_out1=  cv2.resize(img_bin,(int(width),int(height/2)))
        combined_image[int(height/2):int(height),:] =smaller_img_out2
        combined_image[0:int(height/2),:] =smaller_img_out1
        # print(peaks)
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
        cv2.imshow(img_arg+str(count)+".jpg", combined_image)
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
