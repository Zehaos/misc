from pycocotools.coco import COCO
import numpy as np
from lib import pascal_voc_io
import os.path as osp
import cv2

dataDir='/media/zehao/WD/Dataset/processed/COCO'
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
                b_x = (bbox[0] + 0.5*bbox[2])/float(2*img_w)
                b_y = (bbox[1] + 0.5*bbox[3])/float(2*img_h)
                b_w = (bbox[2])/float(img_w)
                b_h = (bbox[3])/float(img_h)
                f.write(cls + ' ' + str(b_x) + ' ' + str(b_y) + ' ' + str(b_w) + ' ' + str(b_h) + '\n')
    cv2.imwrite(osp.join(timgDir, img['file_name']), I)



