import cv2
import os.path as osp
import os

video_path = '/home/zehao/Dataset/kunmin_station/2017-03-13'
timg_path = '/home/zehao/Dataset/kunmin_station/2017-03-13_images'
if not os.path.exists(timg_path):
    os.mkdir(timg_path)
# for each video file
filelist = os.listdir(video_path)

for file in filelist:
    if osp.splitext(file)[1] == '.mp4':

        filename = osp.splitext(file)[0]
        print os.path.join(video_path, file)
        videoCapture = cv2.VideoCapture(os.path.join(video_path, file))

        success, frame = videoCapture.read()
        idx = 0
        f_idx = 0

        while success:
            success, frame = videoCapture.read()
            f_idx += 1
            if f_idx%500==0:
                cv2.imwrite(os.path.join(timg_path, filename+str(idx)+'.jpg'), frame)
                idx += 1
