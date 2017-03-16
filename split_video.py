# -*- coding: utf-8 -*-
import cv2
import os.path as osp
import os
import time
video_path = '/home/zehao/Dataset/kunmin_station/2017-03-16'
timg_path = '/home/zehao/Dataset/kunmin_station/2017-03-16_images'
if not os.path.exists(timg_path):
    os.mkdir(timg_path)

# for each video file
#filelist = os.listdir(video_path)

for dir, subdir, files in os.walk(video_path):
    for file in files:
        if osp.splitext(file)[1] in ['.mp4', '.avi', '.MP4', '.AVI']:

            filename = osp.splitext(file)[0]
            print os.path.join(dir, file)
            videoCapture = cv2.VideoCapture(os.path.join(dir, file))

            success, frame = videoCapture.read()
            idx = 0
            f_idx = 0

            while success:
                success, frame = videoCapture.read()
                f_idx += 1
                if f_idx % 500 == 0:
                    cv2.imwrite(os.path.join(timg_path, str(time.time()) + '.jpg'), frame)
                    idx += 1


