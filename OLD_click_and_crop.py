# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
# import argparse
import cv2
import function_parameters as FP
import Image_processing_functions as IPF
import numpy as np
import copy
import text_print_functions as TPF
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
roi_show=False

xposition=80
yposition=20
# position=(20,80)
font_size=0.5
thickness=1

reset_flag=False
sharpen_flag=False
showimg_flag=False

def show_defined_roi(image,src,w):
	xposition=80
	cv2.putText(image, 'pt0: '+str(src[0]), (xposition,yposition), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
	xposition=80+150
	cv2.putText(image, 'pt1: '+str(src[1]), (xposition,yposition), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
	xposition=80+300
	cv2.putText(image, 'pt2: '+str(src[2]), (xposition,yposition), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
	xposition=80+450
	cv2.putText(image, 'pt3: '+str(src[3]), (xposition,yposition), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)

	cv2.circle(image,(src[0][0],src[0][1]), 4, (0,0,255), -1)
	cv2.circle(image,(src[1][0],src[1][1]), 4, (0,0,255), -1)
	cv2.circle(image,(src[2][0],src[2][1]), 4, (0,0,255), -1)
	cv2.circle(image,(src[3][0],src[3][1]), 4, (0,0,255), -1)

	cv2.line(image,(0,src[0][1]),(w,src[0][1]),(255,0,0),1)
	cv2.line(image,(0,src[2][1]),(w,src[2][1]),(255,0,0),1)


def draw_roi_information(refPt,dst):
	print('+'+'_'*(TPF.line_length-2)+'+')

	print(TPF.print_line_text_in_middle('Calculated ROI',TPF.line_length-2))

	src_text=' SRC => ['+str(refPt[0])+', '+str(refPt[1])+', '+str(refPt[2])+', '+str(refPt[3])+']'
	print(TPF.print_line_text_in_middle(src_text,TPF.line_length-2))
	print('+'+'_'*(TPF.line_length-2)+'+')

	dst_text=' DST => ['+str(dst[0])+', '+str(dst[1])+', '+str(dst[2])+', '+str(dst[3])+']'
	print(TPF.print_line_text_in_middle(dst_text,TPF.line_length-2))
	print('+'+'_'*(TPF.line_length-2)+'+')

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt,xposition,yposition,font_size,thickness

	global image, image_name, w, reset_flag
	# image=param[0]
	# image_name=param[1]
	# w=param[2]


	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if len(refPt) is 4:
		return

	else:



		if event == cv2.EVENT_LBUTTONDOWN:
			# refPt = [(x, y)]
			if len(refPt) is 1 or len(refPt) is 3:
				refPt.append((x, refPt[-1][1]))
			else:
				refPt.append((x, y))
			# print(refPt)
			# cropping = True

			if reset_flag is True:
				image = clone.copy()
				reset_flag=False

		# check to see if the left mouse button was released
		# elif event == cv2.EVENT_LBUTTONUP:
		# 	# record the ending (x, y) coordinates and indicate that
		# 	# the cropping operation is finished
		# 	refPt.append((x, y))
		# 	cropping = False

			if len(refPt) is 1:
				xposition=80
				# yposition=60
			elif len(refPt) is 2:
				xposition=80+150
				# yposition=100
			elif len(refPt) is 3:
				xposition=80+150+150
				# yposition=140
			elif len(refPt) is 4:
				xposition=80+150+150+150
				# yposition=180


			# draw a rectangle around the region of interest
			# cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.putText(image, 'pt'+str(len(refPt))+': '+str(refPt[-1]), (xposition,yposition), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
			cv2.circle(image,(refPt[-1][0],refPt[-1][1]), 4, (0,0,255), -1)
			cv2.line(image,(0,refPt[-1][1]),(w,refPt[-1][1]),(255,0,0),1)
			# cv2.putText(image, str(refPt[-1]), (position[0],position[1]), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)


		# cv2.imshow(img_name, image)
		cv2.imshow(image_name, image)
		# cv2.imshow("image", image)




# def click_and_crop(event, x, y, flags, param):
# 	# grab references to the global variables
# 	global refPt,xposition,yposition,font_size,thickness
#
# 	# if the left mouse button was clicked, record the starting
# 	# (x, y) coordinates and indicate that cropping is being
# 	# performed
# 	if len(refPt) is 4:
# 		return
#
# 	else:
# 		if event == cv2.EVENT_LBUTTONDOWN:
# 			# refPt = [(x, y)]
# 			if len(refPt) is 1 or len(refPt) is 3:
# 				refPt.append((x, refPt[-1][1]))
# 			else:
# 				refPt.append((x, y))
# 			# print(refPt)
# 			# cropping = True
#
# 		# check to see if the left mouse button was released
# 		# elif event == cv2.EVENT_LBUTTONUP:
# 		# 	# record the ending (x, y) coordinates and indicate that
# 		# 	# the cropping operation is finished
# 		# 	refPt.append((x, y))
# 		# 	cropping = False
#
# 			if len(refPt) is 1:
# 				xposition=80
# 				yposition=60
# 			elif len(refPt) is 2:
# 				xposition=80
# 				yposition=100
# 			elif len(refPt) is 3:
# 				xposition=80
# 				yposition=140
# 			elif len(refPt) is 4:
# 				xposition=80
# 				yposition=180
#
# 			# draw a rectangle around the region of interest
# 			# cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
# 			cv2.putText(image, 'pt'+str(len(refPt))+': '+str(refPt[-1]), (xposition,yposition), cv2.FONT_HERSHEY_DUPLEX, font_size, (255,0,255), thickness, cv2.LINE_AA)
# 			cv2.circle(image,(refPt[-1][0],refPt[-1][1]), 3, (0,255,255), -1)
# 			cv2.line(image,(0,refPt[-1][1]),(w,refPt[-1][1]),(0,0,255),1)
# 			# cv2.putText(image, str(refPt[-1]), (position[0],position[1]), cv2.FONT_HERSHEY_DUPLEX, font_size, (0,0,255), thickness, cv2.LINE_AA)
#
#
# 		# cv2.imshow(img_name, image)
# 		cv2.imshow(image_name, image)
# 		# cv2.imshow("image", image)



	# # if the left mouse button was clicked, record the starting
	# # (x, y) coordinates and indicate that cropping is being
	# # performed
	# if event == cv2.EVENT_LBUTTONDOWN:
	# 	refPt = [(x, y)]
	# 	cropping = True
	#
	# # check to see if the left mouse button was released
	# elif event == cv2.EVENT_LBUTTONUP:
	# 	# record the ending (x, y) coordinates and indicate that
	# 	# the cropping operation is finished
	# 	refPt.append((x, y))
	# 	cropping = False
	#
	# 	# draw a rectangle around the region of interest
	# 	cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
	# 	cv2.imshow("image", image)

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

count = FP.frame
dashcam_image_path = FP.dashcam_image_path
img_arg="frame"
image = None
h=0
w=0
# clone = image.copy()
# cv2.namedWindow("image")
image_name=img_arg+str(count)
# param=[image,image_name,w]
# cv2.namedWindow(image_name)
# cv2.setMouseCallback(image_name, click_and_crop,param)


# count = FP.frame
# dashcam_image_path = FP.dashcam_image_path
# img_arg="frame"
#
# image = cv2.imread(dashcam_image_path+img_arg+str(count)+".jpg")
# h,w = image.shape[:2]
# clone = image.copy()
# # cv2.namedWindow("image")
# image_name=img_arg+str(count)
#
# param=[image,image_name,w]
# cv2.namedWindow(image_name)
# cv2.setMouseCallback(image_name, click_and_crop,param)

def main():
	global refPt, roi_show, count, image, image_name, clone, h,w,img_arg, dashcam_image_path,param, reset_flag, sharpen_flag

	count = FP.frame
	dashcam_image_path = FP.dashcam_image_path
	img_arg="frame"
	image = cv2.imread(dashcam_image_path+img_arg+str(count)+".jpg")
	h,w = image.shape[:2]
	clone = image.copy()
	# cv2.namedWindow("image")
	image_name=img_arg+str(count)
	cv2.namedWindow(image_name)
	cv2.setMouseCallback(image_name, click_and_crop)

	# count = FP.frame
	# dashcam_image_path = FP.dashcam_image_path
	# img_arg="frame"

	# load the image, clone it, and setup the mouse callback function
	# image = cv2.imread(dashcam_image_path+img_arg+str(count)+".jpg")
	# h,w = image.shape[:2]
	# clone = image.copy()
	# # cv2.namedWindow("image")
	# image_name=img_arg+str(count)
	# cv2.namedWindow(image_name)

	# param=[image,image_name,w]
	# cv2.setMouseCallback(image_name, click_and_crop,param)
	# # cv2.setMouseCallback(image_name, click_and_crop)
	# # cv2.setMouseCallback("image", click_and_crop)

	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow(image_name, image)
		# cv2.imshow("image", image)


		if len(refPt) == 4 and roi_show is False:
			image2 = clone.copy()
			img_undistort = IPF.undistort(image2)
			src = np.float32([refPt[0],refPt[1],refPt[2],refPt[3]])
			dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
			roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
			cv2.imshow("ROI", roi)
			draw_roi_information(refPt,dst)
			roi_show=True
			# cv2.waitKey(0)
		# if the 'r' key is pressed, reset the cropping region
		# key = cv2.waitKey()
		key = cv2.waitKey(1) & 0xFF



			# if sharpen_flag is False:
			# 	roi_sharpen = roi.copy()
			# 	roi= cv2.blur(roi_sharpen,(5,5))
			# 	cv2.imshow("ROI", roi)
			# 	sharpen_flag=True
			# else:
			# 	image2 = clone.copy()
			# 	img_undistort = IPF.undistort(image2)
			# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
			# 	cv2.imshow("ROI", roi)
			# 	draw_roi_information(src,dst)
			# 	sharpen_flag=False

		# print(key)
		if key == ord("r"):
			# image = clone.copy()
			reset_flag=True
			cv2.destroyWindow("ROI")
			refPt=[]
			roi_show=False

		elif key == ord("s"):
			sharpen_flag = not sharpen_flag

		elif key== ord("1"):
			src = FP.src1
			dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])

			image = clone.copy()
			show_defined_roi(image,src,w)

			image2 = clone.copy()
			img_undistort = IPF.undistort(image2)
			if sharpen_flag is True:
				img_undistort= cv2.blur(img_undistort,(5,5))
			roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
			cv2.imshow("ROI", roi)
			draw_roi_information(src,dst)
			roi_show=True
		#
		# elif key== ord("2"):
		# 	src = FP.src2
		# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
		#
		# 	image = clone.copy()
		# 	show_defined_roi(image,src,w)
		#
		# 	image2 = clone.copy()
		# 	img_undistort = IPF.undistort(image2)
		# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
		# 	cv2.imshow("ROI", roi)
		# 	draw_roi_information(src,dst)
		# 	roi_show=True
		#
		# elif key== ord("3"):
		# 	src = FP.src3
		# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
		#
		# 	image = clone.copy()
		# 	show_defined_roi(image,src,w)
		#
		# 	image2 = clone.copy()
		# 	img_undistort = IPF.undistort(image2)
		# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
		# 	cv2.imshow("ROI", roi)
		# 	draw_roi_information(src,dst)
		# 	roi_show=True
		#
		# elif key== ord("6"):
		# 	src = FP.src6
		# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
		#
		# 	image = clone.copy()
		# 	show_defined_roi(image,src,w)
		#
		# 	image2 = clone.copy()
		# 	img_undistort = IPF.undistort(image2)
		# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
		# 	cv2.imshow("ROI", roi)
		# 	draw_roi_information(src,dst)
		# 	roi_show=True
		# elif key== ord("7"):
		# 	src = FP.src7
		# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
		#
		# 	image = clone.copy()
		# 	show_defined_roi(image,src,w)
		#
		# 	image2 = clone.copy()
		# 	img_undistort = IPF.undistort(image2)
		# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
		# 	cv2.imshow("ROI", roi)
		# 	draw_roi_information(src,dst)
		# 	roi_show=True
		#
		# elif key== ord("8"):
		# 	src = FP.src8
		# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
		#
		# 	image = clone.copy()
		# 	show_defined_roi(image,src,w)
		#
		# 	image2 = clone.copy()
		# 	img_undistort = IPF.undistort(image2)
		# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
		# 	cv2.imshow("ROI", roi)
		# 	draw_roi_information(src,dst)
		# 	roi_show=True
		#
		# elif key== ord("9"):
		# 	src = FP.src9
		# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
		#
		# 	image = clone.copy()
		# 	show_defined_roi(image,src,w)
		#
		# 	image2 = clone.copy()
		# 	img_undistort = IPF.undistort(image2)
		# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
		# 	cv2.imshow("ROI", roi)
		# 	draw_roi_information(src,dst)
		# 	roi_show=True
		#
		# elif key== ord("0"):
		# 	src = FP.src0
		# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
		#
		# 	image = clone.copy()
		# 	show_defined_roi(image,src,w)
		#
		# 	image2 = clone.copy()
		# 	img_undistort = IPF.undistort(image2)
		# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
		# 	cv2.imshow("ROI", roi)
		# 	draw_roi_information(src,dst)
		# 	roi_show=True

		elif key != 255:
			break

		if showimg_flag is True:

			showimg_flag=False

		# else:
			# break



	# # keep looping until the 'q' key is pressed
	# while True:
	# 	# display the image and wait for a keypress
	# 	cv2.imshow("image", image)
	# 	key = cv2.waitKey(1) & 0xFF
	#
	# 	# if the 'r' key is pressed, reset the cropping region
	# 	if key == ord("r"):
	# 		image = clone.copy()
	# 		refPt=[]
	#
	# 	# if the 'c' key is pressed, break from the loop
	# 	elif key == ord("c"):
	# 		break

	# if there are two reference points, then crop the region of interest
	# from teh image and display it

	# if len(refPt) == 4:
	# 	image = clone.copy()
	#
	# 	img_undistort = IPF.undistort(image)
	# 	src = np.float32([refPt[0],refPt[1],refPt[2],refPt[3]])
	# 	dst = np.float32([(int(w*0.35),0),(int(w-w*0.35),0),(int(w*0.35),h),(int(w-w*0.35),h)])
	# 	roi, M, Minv = IPF.unwarp(img_undistort, src, dst)
	# 	cv2.imshow("ROI", roi)
	# 	cv2.waitKey(0)

	# close all open windows
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
