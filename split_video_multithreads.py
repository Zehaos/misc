# -*- coding: utf-8 -*-
import cv2
import os.path as osp
import os
import time
import Queue
import threading

video_path = '/home/zehao/Dataset/kunmin_station/2017-03-21'
timg_path = '/home/zehao/Dataset/kunmin_station/2017-03-21_images'
if not os.path.exists(timg_path):
    os.mkdir(timg_path)

filequeue = Queue.Queue(maxsize=0)

for dir, subdir, files in os.walk(video_path):
    for file in files:
        if osp.splitext(file)[1] in ['.mp4', '.avi', '.MP4', '.AVI']:
          filequeue.put(osp.join(dir, file))


def BreakVideo():
  while(not filequeue.empty()):
    file = filequeue.get()
    print("Processing " + file + '\n')
    videoCapture = cv2.VideoCapture(file)
    success, frame = videoCapture.read()
    idx = 0
    f_idx = 0

    while success:
      success, frame = videoCapture.read()
      f_idx += 1
      if f_idx % 500 == 0:
        cv2.imwrite(os.path.join(timg_path, str(time.time()) + '.jpg'), frame)
        idx += 1

threads = [
  threading.Thread(target=BreakVideo, args=()) for i in xrange(3)
]
for t in threads: t.start()



