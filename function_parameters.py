import numpy as np


#for making video. False makes diagnostic video and True makes only result video
fullscreen=False


video_type='diagnostic'
if fullscreen is True:
    video_type='fullscreen'

video_tip='default_video'

version='v1.0'
#video_name = 'v1.0_rainy_video' + video_type +'.mp4'
#video_name = version+'rainy_better_video' + video_type +'.mp4'
#video_name = 'v1.0_night_video' + video_type +'.mp4'
#video_name = 'v1.5_test_video' + video_type +'.mp4'
#video_name = 'v1.0_foggy_video' + video_type +'.mp4'
#video_name = 'v1.3._challenge_video' + video_type +'.mp4'
# video_name = 'v1.2._harder_challenge_video' + video_type +'.mp4'
#video_name = 'v1.2._project_video' + video_type +'.mp4'

video_name=video_tip+'_'+version+'_'+video_type+'.mp4'
# dashcam_image_path = '/home/profesor/Documents/Datasets/project_video640x360/'
# dashcam_image_path = '/home/profesor/Documents/Datasets/dashcam_driving-640x360/'
# dashcam_image_path = '/home/profesor/Documents/Datasets/dashcam_driving/'
#dashcam_image_path = '/home/profesor/Documents/Datasets/foggy_video/'
#dashcam_image_path = '/home/profesor/Documents/Datasets/night/'
#dashcam_image_path = '/home/profesor/Documents/Datasets/rainy_video_better/'
# dashcam_image_path = '/home/profesor/Documents/Datasets/rainy_video/'
#dashcam_image_path = '/home/profesor/Documents/Datasets/rainy_video2/'
#dashcam_image_path = './Test_images/challnege_video/'
# dashcam_image_path = './Test_images/harder_challenge_video/'
dashcam_image_path = '/home/profesor/Documents/Datasets/challnege_video/'
# dashcam_image_path = '/home/profesor/Documents/Datasets/project_video/'
# dashcam_image_path = '/home/profesor/Documents/Datasets/harder_video2/'

# frame=1
frame=0
calibration_frame=0
image_folder = dashcam_image_path


#binary_combinations=('hls_s','hls_l','lab_b','lab_l','sobel_abs','sobel_mag','sobel_dir',)
binary_combinations=('hsv_white')
# binary_combinations=('rgb_r')
# binary_combinations=('rgb_r','hls_l ')
# binary_combinations=('hls_s','lab_b','hls_l ')
#binary_combinations=('hls_l')
# binary_combinations=('rgb-r')
#binary_combinations=('rgb-r','lab_l','sobel_abs','sobel_mag')
#binary_combinations=('sobel_abs','sobel_mag')
# binary_combinations=()
calibrated_combinations=[]



def unwarp_points(h,w):

    if dashcam_image_path == '/home/profesor/Documents/Datasets/challnege_video/':
        # src = np.float32([(499,466),(781,466),(343,692),(937,692)])
        src = np.float32([(575,464),(707,464),(258,682),(1049,682)])
    elif dashcam_image_path == '/home/profesor/Documents/Datasets/project_video/':
        src = np.float32([(575,464),(707,464),(258,682),(1049,682)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/dashcam_driving/':
        src = np.float32([(int(w*0.43),int(h*0.6)),(int(w*0.57),int(h*0.6)),(int(w*0.16),int(h*0.86)),(int(w*0.84),int(h*0.86))])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/harder_challenge_video/':
        src = np.float32([(529,466),(751,466),(373,692),(907,692)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/night/':
        # src = np.float32([(550,430),(730,430),(200,622),(1080,622)])
         # SRC => [(527, 462), (768, 462), (276, 650), (950, 650)]
        src = np.float32([(529,422),(776,422),(101,600),(1101,600)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/foggy_video/':
        src = np.float32([(431, 437), (791, 437), (256, 647), (889, 647)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/rainy_video/':
        src = np.float32([(550,430),(730,430),(200,622),(1080,622)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/rainy_video_better/':
        src = np.float32([(550,430),(730,430),(200,622),(1080,622)])

    # elif dashcam_image_path == '/home/profesor/Documents/Datasets/rainy_video2/':

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/dashcam_driving-640x360/':
        src = np.float32([(int(w*0.43),int(h*0.6)),(int(w*0.57),int(h*0.6)),(int(w*0.16),int(h*0.86)),(int(w*0.84),int(h*0.86))])

    # elif dashcam_image_path == '/home/profesor/Documents/Datasets/project_video640x360/':

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/spoljar_mrak/':
        src = np.float32([(550,430),(730,430),(200,622),(1080,622)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/spoljar_sunce/':
        src = np.float32([(499,396),(781,396),(343,622),(937,622)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/Grand Canyon/':
        src = np.float32([(499,466),(781,466),(343,692),(937,692)])

    elif dashcam_image_path == '/home/profesor/Documents/Datasets/harder_video2/':
        src = np.float32([(575,444),(707,444),(208,710),(1000,710)])
    elif dashcam_image_path == '/home/profesor/Documents/Datasets/angels_forest_highway/':
        src = np.float32([(550,430),(730,430),(200,622),(1080,622)])
    elif dashcam_image_path == '/home/profesor/Documents/Datasets/glenshane_snowy/':
        src = np.float32([(467, 536), (728, 536), (167, 698), (943, 698)])
    elif dashcam_image_path == '/home/profesor/Documents/Datasets/double_solid_white/':
        src = np.float32([(529, 466), (751, 466), (373, 692), (907, 692)])
    elif dashcam_image_path == '/home/profesor/Documents/Datasets/toronto_wet_drive/':
        src = np.float32([(529, 464), (776, 464), (101, 682), (1101, 682)])
    elif dashcam_image_path == '/home/profesor/Documents/Datasets/night_highway/':
        src = np.float32([(529, 396), (751, 396), (373, 622), (907, 662)])





    else:
        # src = np.float32([(575,464),
        #                   (707,464),
        #                   (258,682),
        #                   (1049,682)])
        # src = np.float32([(300,232),
        #                   (350,232),
        #                   (124,341),
        #                   (520,341)])
        src = np.float32([(int(w*0.45),int(h*0.6)),
                              (int(w*0.57),int(h*0.6)),
                              (int(w*0.16),int(h*0.86)),
                              (int(w*0.84),int(h*0.86))])
        # src = np.float32([(int(w*0.43),int(h*0.6)),
        #                       (int(w*0.57),int(h*0.6)),
        #                       (int(w*0.16),int(h*0.86)),
        #                       (int(w*0.84),int(h*0.86))])


    dst = np.float32([(int(w*0.35),0),
                          (int(w-w*0.35),0),
                          (int(w*0.35),h),
                          (int(w-w*0.35),h)])
    # dst = np.float32([(450,0),
    #                       (w-450,0),
    #                       (450,h),
    #                       (w-450,h)])
    # dst = np.float32([(225,0),
    #                       (w-225,0),
    #                       (225,h),
    #                       (w-225,h)])
    return src,dst




#src constants for clip_and_crop.py program
src1 = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])
src2 = np.float32([(550,430),
                      (730,430),
                      (200,622),
                      (1080,622)])
src3 = np.float32([(529,464),
                      (710,464),
                      (258,682),
                      (1049,682)])
src4 = np.float32([(529,464),
                      (776,464),
                      (101,682),
                      (1101,682)])
src6 = np.float32([(449,396),
                      (831,396),
                      (297,622),
                      (987,622)])
src7 = np.float32([(499,396),
                      (781,396),
                      (343,622),
                      (937,622)])
src8 = np.float32([(529,396),
                      (751,396),
                      (373,622),
                      (907,622)])
src9 = np.float32([(529,466),
                      (751,466),
                      (373,692),
                      (907,692)])
src0 = np.float32([(499,466),
                      (781,466),
                      (343,692),
                      (937,692)])

# src3 = np.float32([(int(w*0.43),int(h*0.6)),
#                     (int(w*0.57),int(h*0.6)),
#                     (int(w*0.16),int(h*0.86)),
#                     (int(w*0.84),int(h*0.86))])
