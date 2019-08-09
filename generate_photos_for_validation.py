import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
import Image_processing_functions as IPF
import sys
import function_parameters as FP
import Lane_find_functions as LFF
import time
# from Lane_find_functions import *

def calculate_IOU_all_lines_considered(processed_image,processed_lane_inds,gt_image,gt_lane_inds):
    iou_calculations=[None,None,None,None]

    rows,cols,_ = gt_image.shape

    print('gt_lane_inds='+str(gt_lane_inds))
    for x,element in enumerate(gt_lane_inds):
        if element != str(0):
            zajednicko=0
            unija=0
            gt_pixels=[]
            processed_pixels=[]
            for i in range(rows):
                for j in range(cols):
                    if processed_image[i,j,0] != 0:
                        processed_pixels.append((i,j))
                    if gt_image[i,j,0] == x+1:
                        gt_pixels.append((i,j))
            # print('processed_pixels['+str(x)+']: '+str(processed_pixels))
            # print('gt_pixels['+str(x)+']: '+str(gt_pixels))
            print('length of processed_pixels['+str(x)+']: '+str(len(processed_pixels)))
            print('length of gt_pixels['+str(x)+']: '+str(len(gt_pixels)))
            unija=len(processed_pixels)+len(gt_pixels)
            for j,position in enumerate(gt_pixels):
                if position in processed_pixels:
                    zajednicko+=1
            print('unija='+str(unija))
            print('zajednicko='+str(zajednicko))
            if unija is not 0:
                print('zajednicko/unija='+str(unija)+'/'+str(zajednicko)+'='+str(zajednicko/unija))
                iou_calculations[x]=zajednicko/unija
            else:
                print('zajednicko/unija=0')
                iou_calculations[x]=0.0
        else:
            iou_calculations[x]='x'
    return iou_calculations

def calculate_IOU_single_line_considered(processed_image,processed_lane_inds,gt_image,gt_lane_inds):
    iou_calculations=[None,None,None,None]

    rows,cols,_ = gt_image.shape

    print('gt_lane_inds='+str(gt_lane_inds))
    for x,element in enumerate(gt_lane_inds):
        if element != str(0):
            zajednicko=0
            unija=0
            gt_pixels=[]
            processed_pixels=[]
            for i in range(rows):
                for j in range(cols):
                    if processed_image[i,j,0] == x+1:
                        processed_pixels.append((i,j))
                    if gt_image[i,j,0] == x+1:
                        gt_pixels.append((i,j))
            # print('processed_pixels['+str(x)+']: '+str(processed_pixels))
            # print('gt_pixels['+str(x)+']: '+str(gt_pixels))
            print('length of processed_pixels['+str(x)+']: '+str(len(processed_pixels)))
            print('length of gt_pixels['+str(x)+']: '+str(len(gt_pixels)))
            unija=len(processed_pixels)+len(gt_pixels)
            for j,position in enumerate(gt_pixels):
                if position in processed_pixels:
                    zajednicko+=1
            print('unija='+str(unija))
            print('zajednicko='+str(zajednicko))
            if unija is not 0:
                print('zajednicko/unija='+str(unija)+'/'+str(zajednicko)+'='+str(zajednicko/unija))
                iou_calculations[x]=zajednicko/unija
            else:
                print('zajednicko/unija=0')
                iou_calculations[x]=0.0
        else:
            iou_calculations[x]='x'



    # zajednicko=0
    # unija=0
    # gt_pixels=[]
    # processed_pixels=[]
    # for i in range(rows):
    #     for j in range(cols):
    #         if processed_image[i,j,0] !=0:
    #             processed_pixels.append((i,j))
    #         if gt_image[i,j,0] != 0:
    #             gt_pixels.append((i,j))
    # # print('processed_pixels['+str(x)+']: '+str(processed_pixels))
    # # print('gt_pixels['+str(x)+']: '+str(gt_pixels))
    # print('length of processed_pixels['+str(x)+']: '+str(len(processed_pixels)))
    # print('length of gt_pixels['+str(x)+']: '+str(len(gt_pixels)))
    # unija=len(processed_pixels)+len(gt_pixels)
    # for j,position in enumerate(gt_pixels):
    #     if gt_image[position[0],position[1],0] == j+1
    #     if position in processed_pixels:
    #         zajednicko+=1
    # print('unija='+str(unija))
    # print('zajednicko='+str(zajednicko))
    # if unija is not 0:
    #     print('zajednicko/unija='+str(unija)+'/'+str(zajednicko)+'='+str(zajednicko/unija))
    #     iou_calculations[x]=zajednicko/unija
    # else:
    #     print('zajednicko/unija=0')
    #     iou_calculations[x]=0.0


    return iou_calculations

def main():


    ts = time.gmtime()
    readable_time = time.strftime("%Y-%m-%d", ts)
    print('time is: '+str(readable_time))

    prefix='/home/profesor/Documents/CUlane/data/CULane'
    save_folder='/home/profesor/Documents/advanced_lane_lanes/validation_images/'
    # f = open("val_gt.txt", "r")
    f = open("validation_groundtruth_new.txt", "r")

    image_location=''

    for line in f:
    # line=f.readline()


        # line=f.readline()
        print(line)
        parsed_line=line.split()
        image_location=prefix+parsed_line[0]
        print(image_location)

        file_name_array=parsed_line[0].split('/')
        file_name=file_name_array[3]
        folder_name=file_name_array[2]
        print(file_name)

        compare_folder_path=prefix+parsed_line[1]


        imgOriginal = cv2.imread(image_location)
        if imgOriginal is None:                             # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return


        img_undistort = IPF.undistort(imgOriginal)
        #get points on image for perspective transform
        h,w = imgOriginal.shape[:2]
        src,dst = FP.unwarp_points(h,w)

        # Perspective Transform
        img_unwarp, M, Minv = IPF.unwarp(img_undistort, src, dst)

        first=IPF.calibrate_IPF_all(img_unwarp)
        FP.binary_combinations=(first[0][1],first[1][1])
        # print('_-_-_-_-binary combinations_-_-_-_-'+str(first[0][1])+'---'+str(first[1][1]))


        new_save_folder_path=save_folder+folder_name+'/'+file_name
        # compare_folder_path=save_folder+folder_name+'/'+file_name
        save_folder_path = new_save_folder_path.replace(".jpg", ".png")

        processed_image,lane_count =LFF.process_image_for_validation(imgOriginal,save_folder_path,compare_folder_path,fullscreen=False)

        gt_image=cv2.imread(compare_folder_path)
        if gt_image is None:                             # if image was not read successfully
            print ("error: image not read from file \n\n")        # print error message to std out
            os.system("pause")                                  # pause so user can see error message
            return
        cv2.imwrite(save_folder_path+'_gt',gt_image)

        gt_lane_inds=parsed_line[2]+' '+parsed_line[3]+' '+parsed_line[4]+' '+parsed_line[5]
        gt_lane_inds_field=[parsed_line[2],parsed_line[3],parsed_line[4],parsed_line[5]]


        iou_calculations_signle_line=calculate_IOU_single_line_considered(processed_image,lane_count,gt_image,gt_lane_inds_field)
        iou_calculations_all_line=calculate_IOU_all_lines_considered(processed_image,lane_count,gt_image,gt_lane_inds_field)
        print(iou_calculations_signle_line)
        print(iou_calculations_all_line)
        # completeName = os.path.join(save_folder+folder_name+'/', "validation-"+str(readable_time)+".txt")
        completeName_single_line = os.path.join(save_folder+folder_name+'/', "validation-"+str(readable_time)+".txt")
        ff= open(completeName_single_line,"a")
        print(completeName_single_line)
        completeName_all_lines = os.path.join(save_folder+folder_name+'/', "validation-All-Lines-Considered-"+str(readable_time)+".txt")
        fff= open(completeName_all_lines,"a")
        completeName_in_one_folder = "validation-selected-result-"+str(readable_time)+".txt"
        ffff= open(completeName_in_one_folder,"a")
        print(completeName_all_lines)
        # newcompare_folder_path=compare_folder_path.replace(prefix,'')
        # newsave_folder_path=save_folder_path.replace(save_folder,'')
        # write_line=newcompare_folder_path+' '+newsave_folder_path+' '+gt_lane_inds+' | '+str(lane_count[0])+' '+str(lane_count[1])+' '+str(lane_count[2])+' '+str(lane_count[3])
        write_line_single_line=compare_folder_path+' '+save_folder_path+' '+gt_lane_inds+' | '+str(lane_count[0])+' '+str(lane_count[1])+' '+str(lane_count[2])+' '+str(lane_count[3])+' | '+str(iou_calculations_signle_line[0])+' '+str(iou_calculations_signle_line[1])+' '+str(iou_calculations_signle_line[2])+' '+str(iou_calculations_signle_line[3])+'\n'
        print(write_line_single_line)
        write_lineall_line=compare_folder_path+' '+save_folder_path+' '+gt_lane_inds+' | '+str(lane_count[0])+' '+str(lane_count[1])+' '+str(lane_count[2])+' '+str(lane_count[3])+' | '+str(iou_calculations_all_line[0])+' '+str(iou_calculations_all_line[1])+' '+str(iou_calculations_all_line[2])+' '+str(iou_calculations_all_line[3])+'\n'
        print(write_lineall_line)

        ff.write(write_line_single_line)
        fff.write(write_lineall_line)
        ffff.write(write_lineall_line)



    f.close()
    ff.close()
    fff.close()
    ffff.close()
    return

###################################################################################################
if __name__ == "__main__":
    main()
