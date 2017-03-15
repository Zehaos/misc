import numpy as np
from lib import pascal_voc_io
import os.path as osp
import cv2

imglist = '/home/zehao/Dataset/kunmin_station/2017-03-13_imgs_list.txt'
img_dir = '/home/zehao/Dataset/kunmin_station/2017-03-13_images'
label_dir = '/home/zehao/Dataset/kunmin_station/2017-03-13_images'
tlabelDir = '/home/zehao/Dataset/kunmin_station/2017-03-13_images'

with open(imglist, 'r') as f:
    lines = f.readlines()



for line in lines:
    voc_writer = pascal_voc_io.PascalVocWriter(tlabelDir, None, None)
    img_name = osp.basename(line).strip()
    label_name = img_name[:-4] + '.txt'
    with open(osp.join(label_dir, label_name)) as f:
        objects = f.readlines()
        if len(objects) == 0:
            continue
        for obj in objects:
            x = float(obj.split(' ')[1])
            y = float(obj.split(' ')[2])
            w = float(obj.split(' ')[3])
            h = float(obj.split(' ')[4])
            img = cv2.imread(osp.join(img_dir, img_name))
            img_size = np.shape(img)
            img_height = img_size[0]
            img_width = img_size[1]
            img_channel = img_size[2]

            i_w = img_width*w
            i_h = img_height*h
            i_x = img_width * x - 0.5*i_w
            i_y = img_height * y - 0.5*i_h

            voc_writer.imgSize = [img_height, img_width, img_channel]
            voc_writer.filename = img_name
            voc_writer.foldername = 'labels'
            voc_writer.localImgPath = osp.join(img_dir, img_name)
            voc_writer.addBndBox(int(i_x), int(i_y), int(i_x+i_w), int(i_y+i_h), 'person')
        print(osp.join(tlabelDir, img_name[:-4] + '.xml'))
        voc_writer.save(osp.join(tlabelDir, img_name[:-4] + '.xml'))



