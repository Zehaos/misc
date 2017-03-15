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
    voc_writer = pascal_voc_io.PascalVocWriter(tlabelDir, None, None)
    for ann in anns:
        if ann['category_id'] == catIds[0]:
            bbox = ann['bbox']
            name = 'person'
            voc_writer.imgSize = [np.shape(I)[0], np.shape(I)[1], np.shape(I)[2]]
            voc_writer.filename = img['file_name']
            voc_writer.foldername = 'labels'
            voc_writer.localImgPath = osp.join(timgDir, img['file_name'])
            voc_writer.addBndBox(bbox[0],bbox[1],bbox[2],bbox[3],name)
    #cv2.imwrite(osp.join(timgDir, img['file_name']),I)
    voc_writer.save(osp.join(tlabelDir, img['file_name'][:-4]+'.xml'))


