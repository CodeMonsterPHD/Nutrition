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
    img_file = os.path.join('../ECUSTFD/ECUSTFD-resized--master/JPEGImages')
    img_list = os.listdir('../ECUSTFD/ECUSTFD-resized--master/JPEGImages')
    after_train = []
    with open("../../ECUSTFD-resized--master/ImageSets/Main/trainval.txt", "r") as f:  # 打开文件
        data_train = f.read()  # 读取文件
        for i in img_list:
            u,v = i.split('.')
            if(u in data_train):
                 after_train.append(i)
    after_test = []
    with open("../../ECUSTFD-resized--master/ImageSets/Main/test.txt", "r") as f:  # 打开文件
        data_test = f.read()  # 读取文件
        for i in img_list:
            u, v = i.split('.')
            if (u in data_test):
                 after_test.append(i)

    for i in after_train:
        img_path = os.path.join(img_file, i)
        img_array=cv2.imread(img_path,cv2.IMREAD_COLOR)
        # '''调用cv2.resize函数resize图片'''
        new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        # '''调用cv.2的imwrite函数保存图片'''
        cv2.imwrite(os.path.join('../../ECUSTFD-resized--master/TrainImage', i), new_array)
    for i in after_test:
        img_path = os.path.join(img_file, i)
        img_array=cv2.imread(img_path,cv2.IMREAD_COLOR)
        # '''调用cv2.resize函数resize图片'''
        new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        # '''调用cv.2的imwrite函数保存图片'''
        cv2.imwrite(os.path.join('../../ECUSTFD-resized--master/TestImage', i), new_array)
if __name__ == '__main__':
    main()