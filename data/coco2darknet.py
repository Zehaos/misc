from pycocotools.coco import COCO
import numpy as np
from lib import pascal_voc_io
import os.path as osp
import cv2
import random
dataDir='/home/zehao/Dataset/COCO'
dataType='train2014'
annFile='%s/annotations/instances_%s.json'%(dataDir, dataType)

timgDir='/home/zehao/Dataset/coco_person/images'
tlabelDir='/home/zehao/Dataset/coco_person/labels'

coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print 'COCO categories: \n\n', ' '.join(nms)

nms = set([cat['supercategory'] for cat in cats])
print 'COCO supercategories: \n', ' '.join(nms)

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

for imgId in imgIds:
    img = coco.loadImgs(imgId)[0]
    I = cv2.imread('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    with open(osp.join(tlabelDir, osp.splitext(img['file_name'])[0]+'.txt'), 'w') as f:
        for ann in anns:
            if ann['category_id'] == catIds[0]:
                bbox = ann['bbox']
                cls = '0'
                img_shape = np.shape(I)
                img_h = img_shape[0]
                img_w = img_shape[1]
                b_x = (bbox[0] + 0.5*bbox[2])/float(img_w)
                b_y = (bbox[1] + 0.5*bbox[3])/float(img_h)
                b_w = (bbox[2])/float(img_w)
                b_h = (bbox[3])/float(img_h)
                """
                x_ = b_x*img_w
                y_ = b_y*img_h
                w_ = b_w*img_w
                h_ = b_h*img_h
                x_min = x_ - 0.5 * w_
                x_max = x_ + 0.5 * w_
                y_min = y_ - 0.5 * h_
                y_max = y_ + 0.5 * h_
                cv2.rectangle(I, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 2)
                cv2.rectangle(I, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,255), 2)
                cv2.imshow('show', I)
                cv2.waitKey(0)
                """
                f.write(cls + ' ' + str(b_x) + ' ' + str(b_y) + ' ' + str(b_w) + ' ' + str(b_h) + '\n')
    #cv2.imwrite(osp.join(timgDir, img['file_name']), I)



