import cv2
import os.path as osp
import os
import Queue
import threading

image_path = '/home/zehao/Dataset/KITTI/object/training/image_2'
target_path = '/home/zehao/Dataset/KITTI/object/training/image_2'

ori_format = '.png'
target_format = '.jpg'

filequeue = Queue.Queue(maxsize=0)

for root, dirs, files in os.walk(image_path):
  for file in files:
    file_name = osp.splitext(file)[0]
    filequeue.put(file_name)

def format_conversion():
  while(not filequeue.empty()):
    file_name = filequeue.get()
    print("Processing " + file_name + '\n')
    img = cv2.imread(osp.join(root, file_name + ori_format))
    cv2.imwrite(osp.join(target_path, file_name + target_format), img)


threads = [
  threading.Thread(target=format_conversion, args=()) for i in xrange(3)
]
for t in threads: t.start()