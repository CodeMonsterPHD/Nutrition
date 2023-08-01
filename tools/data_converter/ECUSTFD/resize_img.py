import os
import shutil

import cv2
import shutil

IMG_SIZE= 256

def main():
    img_file = os.path.join('')
    img_list = os.listdir('')
    after_train = []
    with open("trainval.txt", "r") as f:
        data_train = f.read()
        for i in img_list:
            u,v = i.split('.')
            if(u in data_train):
                 after_train.append(i)
    after_test = []
    with open("test.txt","r") as f:
        data_test = f.read()
        for i in img_list:
            u, v = i.split('.')
            if (u in data_test):
                 after_test.append(i)

    for i in after_train:
        img_path = os.path.join(img_file, i)
        img_array=cv2.imread(img_path,cv2.IMREAD_COLOR)
        new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        cv2.imwrite(os.path.join('TrainImage',i),new_array)
    for i in after_test:
        img_path = os.path.join(img_file, i)
        img_array=cv2.imread(img_path,cv2.IMREAD_COLOR)
        new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        cv2.imwrite(os.path.join('TestImage',i),new_array)

if __name__ == '__main__':
    main()