import random

with open('/home/zehao/Dataset/Caltech/images.txt', 'r') as f:
    lines = f.readlines()

num = len(lines)

val_num = int(num*0.5)
train_num = num - val_num

random.shuffle(lines)

val_list = lines[0:val_num]
train_list = lines[val_num:num]

with open('/home/zehao/Dataset/Caltech/val.txt', 'w') as f:
    f.writelines(val_list)

with open('/home/zehao/Dataset/Caltech/train.txt', 'w') as f:
    f.writelines(train_list)