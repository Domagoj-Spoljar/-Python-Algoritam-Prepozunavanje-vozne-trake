import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Lane_find_functions as Lff

mtx = np.array([[1.15694035e+03, 0.00000000e+00, 6.65948597e+02],[0.00000000e+00, 1.15213869e+03, 3.88785178e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],np.float64)
dist = np.array([-2.37636612e-01, -8.54129776e-02, -7.90955950e-04, -1.15908700e-04, 1.05741395e-01],np.float64)

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
        exampleImg_undistort = Lff.undistort(exampleImg)
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

        exampleImg_unwarp, M, Minv = Lff.unwarp(exampleImg_undistort, src, dst)

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
        exampleImg_sobelAbs = Lff.abs_sobel_thresh(exampleImg_unwarp, 'x', min_thresh, max_thresh)

        #--------------------------------------------------------------
        min_thresh2=30
        max_thresh2=200
        kernel_size=(1,31,2)
        exampleImg_sobelMag = Lff.mag_thresh(exampleImg_unwarp, kernel_size, (min_thresh2, max_thresh2))

        #--------------------------------------------------------------
        min_thresh3=0
        max_thresh3=0.09
        kernel_size2=7
        exampleImg_sobelDir = Lff.dir_thresh(exampleImg_unwarp, kernel_size2, (min_thresh3, max_thresh3))

        combined = np.zeros_like(exampleImg_sobelMag)
        combined[((exampleImg_sobelMag == 1) & (exampleImg_sobelDir == 1))] = 1

        exampleImg_SThresh = Lff.hls_sthresh(exampleImg_unwarp)
        exampleImg_LThresh = Lff.hls_lthresh(exampleImg_unwarp)
        exampleImg_LBThresh = Lff.lab_bthresh(exampleImg_unwarp)
        #-----------------------------------------------------------------
        combined_HLSl_LABb = np.zeros_like(exampleImg_LBThresh)
        combined_HLSl_LABb[((exampleImg_LBThresh == 1) | (exampleImg_LThresh == 1))] = 1

        combined_HLSs_LABb = np.zeros_like(exampleImg_SThresh)
        combined_HLSs_LABb[((exampleImg_LBThresh == 1) | (exampleImg_SThresh == 1))] = 1



        #cv2.imshow(img_arg+str(count)+".jpg", copy_image)
        #cv2.imshow(img_arg+str(count)+".jpg", resized_img)
        rows,cols,channels = imgOriginal.shape
        veca_slika=np.zeros_like(imgOriginal)

        small_org=cv2.resize(imgOriginal,(int(cols/3)*2,int(rows/3)*2))
        redd,stupp,kanall = small_org.shape
        veca_slika[rows-redd:rows,0:stupp]=small_org
        #print(small_org.shape)

        # small_org=cv2.resize(imgOriginal,(int(cols/3),int(rows/3)))
        # red,stup,kanal = small_org.shape
        # veca_slika[0:red,0:stup]=small_org

        resized_warped=cv2.resize(exampleImg_unwarp,(int(cols/3),int(rows/3)))
        red2,stup2,kanal2 = resized_warped.shape
        veca_slika[0:red2,stup2+stup2:cols-2]=resized_warped
        red,stup,kanal = resized_warped.shape

        # resized_warped=cv2.resize(exampleImg_undistort,(int(cols/3),int(rows/3)))
        # red4,stup4,kanal4 = resized_warped.shape
        # veca_slika[0:red,stup:stup+stup4]=resized_warped

        res=cv2.bitwise_and(exampleImg_unwarp,exampleImg_unwarp, mask=combined_HLSl_LABb)
        resized_warped_lines=cv2.resize(res,(int(cols/3),int(rows/3)))
        red3,stup3,kanal3 = resized_warped_lines.shape
        cv2.line(resized_warped_lines,(0,red3//2),(stup3,red3//2),(0,0,255),2)
        veca_slika[red:red+red3,stup+stup3:cols-2]=resized_warped_lines

        binary_image=combined_HLSl_LABb
        histogram = np.sum(binary_image[binary_image.shape[0]//2:,:], axis=0)
        histogram_image=np.ones((binary_image.shape[0]//2,binary_image.shape[1]),dtype=int)
        #histogram_image=np.ones((binary_image.shape[0]//2,binary_image.shape[1]),dtype=int)
        out_image = np.uint8(np.dstack((histogram_image, histogram_image, histogram_image))*255)
        i=1
        while i <= len(histogram)-1:
            #histogram_image[histogram_image.shape[0]-int(histogram[i]),i]=0
            cv2.line(out_image,(i-1,histogram_image.shape[0]-int(histogram[i-1])),(i,histogram_image.shape[0]-int(histogram[i])),(0,0,0),5)
            i+=1

        resizzeed=np.copy(out_image)
        resizzeed=cv2.resize(resizzeed,(int(cols/3),int(rows/3)))

        # resized_histogram=np.zeros_like(resized_warped_lines)
        # resized_histogram=resizzeed
        # resized_histogram=resizzeed
        # resized_histogram=resizzeed

        #print(resized_histogram.shape)
        red2,stup2, kanal5 = resizzeed.shape
        #veca_slika[red:red+red2,stup+stup2:cols-2]=resizzeed
        veca_slika[red+red2:rows,stup+stup2:cols-2]=resizzeed


        left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = Lff.sliding_window_polyfit(binary_image)
        rectangles = visualization_data[0]
        out_img = np.uint8(np.dstack((binary_image, binary_image, binary_image))*255)
        for rect in rectangles:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)
            cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]


        out_img=Lff.plot_fit_onto_img(out_img,left_fit,(255,0,255))
        out_img=Lff.plot_fit_onto_img(out_img,right_fit,(255,0,255))

        windows_image=cv2.resize(out_img,(int(cols/3),int(rows/3)))
        red6,stup6,kanal6 = windows_image.shape
        veca_slika[0:red6,stup6:stup6*2]=windows_image

        #cv2.imshow('histogram', out_image)
        #cv2.imshow('histogram',resized_histogram)

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
