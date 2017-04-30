import numpy as np
from lib import pascal_voc_io
import os.path as osp
import cv2

imglist = '/home/zehao/Dataset/Gesture/val/color_image_list.txt'
img_dir = '/home/zehao/Dataset/Gesture/val/color_image'
label_dir = '/home/zehao/Dataset/Gesture/val/label'

tlabelDir = '/home/zehao/Dataset/Gesture/val/label'

class_to_type={0:'fist', 1:'good', 2:'index_finger', 3:'yeah', 4:'ok', 5:'palm'}
type_to_class = {'fist':0, 'good':1, 'finger':2, 'yeah':3, 'ok':4, 'palm':5}
with open(imglist, 'r') as f:
    lines = f.readlines()

for line in lines:

    img_name = osp.basename(line).strip()
    img_path = osp.join(img_dir, img_name)
    label_name = img_name[:-4]
    label_path = osp.join(label_dir, label_name + '.xml')

    # read img
    img = cv2.imread(img_path)
    [img_h, img_w, img_c] = np.shape(img)

    if (osp.exists(label_path)):
      voc_reader = pascal_voc_io.PascalVocReader(label_path)
      shapes = voc_reader.getShapes()

      for shape in shapes:
        cls = type_to_class[shape[0]]
        points = shape[1]
        (xmin, ymin) = points[0]
        (xmax, ymax) = points[2]
        with open(osp.join(tlabelDir, label_name + '.txt'), 'w') as f:
          x = (xmin + xmax) * 0.5 / img_w
          y = (ymin + ymax) * 0.5 / img_h
          w = float(xmax - xmin) / img_w
          h = float(ymax - ymin) / img_h
          f.write('%d %f %f %f %f' % (cls, x, y, w, h))
      else:
        f = open(osp.join(tlabelDir, label_name + '.txt'), 'w')
        f.close()




