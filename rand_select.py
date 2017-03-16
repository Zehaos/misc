import random
import os

imgDir = '/home/zehao/Dataset/kunmin_station/2017-03-15_images'
tDir = '/home/zehao/Dataset/kunmin_station/2017-03-15_images_select1000'
if not os.path.exists(tDir):
    os.mkdir(tDir)

file_list = os.listdir(imgDir)
random.shuffle(file_list)

selected = file_list[0:1000]

for file in selected:
    file_path = os.path.join(imgDir, file)
    os.system('mv ' + file_path + ' ' + tDir)

print file_list