import cv2
import os
import numpy as np

img_dir = '/home/zehao/MatlabWorkspace/CaltechTools/data-USA/images'
new_anno_dir = '/home/zehao/MatlabWorkspace/CaltechTools/data-USA/new_annotations/anno_train_1xnew'

t_img_dir = '/home/zehao/Dataset/caltech/darknet_format/imgs'
t_label_dir = '/home/zehao/Dataset/caltech/darknet_format/labels'

for dir, subdir, files in os.walk(img_dir):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':
            video_idx = os.path.split(dir)[1]
            set_idx = os.path.split(os.path.split(dir)[0])[1]
            tfilename = set_idx + '_' + video_idx + '_' + file
            print 'cp '+os.path.join(dir,file)+' '+os.path.join(t_img_dir,tfilename)
            os.system('cp '+os.path.join(dir,file)+' '+os.path.join(t_img_dir,tfilename))

            #read label
            label = os.path.join(new_anno_dir, tfilename+'.txt')
            t_label = os.path.join(t_label_dir, os.path.splitext(tfilename)[0]+'.txt')
            img = cv2.imread(os.path.join(dir, file))
            img_size = np.shape(img)
            img_h = img_size[0]
            img_w = img_size[1]
            img_c = img_size[2]
            with open(label, 'r') as f:
                with open(t_label, 'w') as fout:
                    lines = f.readlines()
                    for line in lines[1:]:
                        cls = line.split(' ')[0]
                        if cls != 'person':
                            continue
                        x = float(line.split(' ')[1])
                        y = float(line.split(' ')[2])
                        w = float(line.split(' ')[3])
                        h = float(line.split(' ')[4])
                        fout.write('0 ' + str((x+0.5*w)/float(img_w)) + ' '
                                         + str((y+0.5*h)/float(img_h)) + ' '
                                         + str(w/float(img_w)) + ' '
                                         + str(h/float(img_h))+'\n')
            print tfilename

