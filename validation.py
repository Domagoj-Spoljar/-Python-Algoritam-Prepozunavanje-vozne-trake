import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import Image_processing_functions as IPF
import sys
import function_parameters as FP

def main():
    # if len(sys.argv) == 2:
    #     img_arg=sys.argv[1]
    #     print('Processing image: '+img_arg)
    # else:
    #     img_arg='frame'+str(FP.frame)
    #     print('Processing default image: '+img_arg)

    img_arg='04830'
    original_image_folder ='06011140_0164.MP4/'
    original_image_path = '/home/profesor/Documents/CUlane/data/CULane/driver_182_30frame/'+original_image_folder
    originalImg = cv2.imread(original_image_path+img_arg+".jpg")
    originalImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)

/driver_182_30frame/06011140_0164.MP4/04830.jpg
/laneseg_label_w16/driver_182_30frame/06011140_0164.MP4/04830.png

    gt_image_path = '/home/profesor/Documents/CUlane/data/CULane/laneseg_label_w16/driver_161_90frame/'+original_image_folder
    gtImg = cv2.imread(gt_image_path+img_arg+".png")
    print(str(gt_image_path+img_arg+".png"))
    # /home/profesor/Documents/CUlane/data/CULane/laneseg_label_w16/driver_161_90frame/06030849_0765.MP4
    # gtImg = cv2.cvtColor(gtImg, cv2.COLOR_BGR2RGB)

    f, ax = plt.subplots(4, 4, figsize=(20,10))
    f.subplots_adjust(hspace = .1, wspace=0.01)

    ax[0,0].imshow(originalImg)
    ax[0,0].axis('off')
    ax[0,0].set_title('Original Image', fontsize=15)
    #
    ax[0,1].imshow(gtImg,cmap='gray')
    ax[0,1].axis('off')
    ax[0,1].set_title('Ground Truth', fontsize=15)

    # ax[0,2].imshow(exampleImg_unwarp)
    # ax[0,2].axis('off')
    # ax[0,2].set_title('Unwarped Image', fontsize=15)


    #------------------------------------------------------
    plt.show()

    return

###################################################################################################
if __name__ == "__main__":
    main()
