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
    img_file = os.path.join('/disk/btc010001/New_Nutrition5K/nutrition5k_dataset/imagery/realsense_overhead')
    img_list = os.listdir('/disk/btc010001/New_Nutrition5K/nutrition5k_dataset/imagery/realsense_overhead')
    for i in img_list:
        if i == 'frames' or i == 'frames_sampled':
            pass
        else:
            iimg_list = os.listdir(os.path.join(img_file, i))
            img_name = os.path.join('/disk/btc010001/New_Nutrition5K/nutrition5k_dataset/imagery/new_realsense_overhead', i)
            os.makedirs(img_name)
            for j in iimg_list:
                if j == 'rgb.png':
                    img_path = os.path.join(img_file, i, j)
                    img_array=cv2.imread(img_path,cv2.IMREAD_COLOR)
                    # '''调用cv2.resize函数resize图片'''
                    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                    # '''调用cv.2的imwrite函数保存图片'''
                    cv2.imwrite(os.path.join(img_name,j),new_array)
if __name__ == '__main__':
    main()