import os
import shutil

import cv2
import shutil
IMG_SIZE= 256

def main():
    img_file = os.path.join('')
    img_list = os.listdir('')
    for j in img_list:
        img_ = os.listdir(os.path.join(img_file,j))
        for i in img_:
            img_path = os.path.join(img_file,j,i)
            img_array=cv2.imread(img_path,cv2.IMREAD_COLOR)
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            cv2.imwrite(os.path.join('vfdl/train_resize',i),new_array)
if __name__ == '__main__':
    main()