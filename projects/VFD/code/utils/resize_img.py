import os
import shutil

import cv2
import shutil
# ''' 设置图片路径，
# '''设置目标像素大小，此处设为256'''
IMG_SIZE= 256

# '''使用os.path模块的join方法生成路径'''
# '''使用os.listdir(path)函数，返回path路径下所有文件的名字，以及文件夹的名字，
def main():
    img_file = os.path.join('../../VFD_dataset/vfdl/train')
    img_list = os.listdir('../../VFD_dataset/vfdl/train')
    for j in img_list:
        img_ = os.listdir(os.path.join(img_file,j))
        for i in img_:
            img_path = os.path.join(img_file,j,i)
            img_array=cv2.imread(img_path,cv2.IMREAD_COLOR)
            # '''调用cv2.resize函数resize图片'''
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            # '''调用cv.2的imwrite函数保存图片'''
            cv2.imwrite(os.path.join('../../VFD_dataset/vfdl/train_resize', i), new_array)



if __name__ == '__main__':
    main()