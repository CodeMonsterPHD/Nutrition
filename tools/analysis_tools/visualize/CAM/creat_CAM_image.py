from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50
import numpy as np
import os
import torchvision.transforms as transforms
from cv2 import cv2
import torch
from PIL import Image
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 1.加载模型
DISH = ['calories','mass','fat','carb','protein']

def cam(model,dish,dish_name):
    target_layer = model.layer4[-1]
    image_path = '/disk/btc010001/New_Nutrition5K/nutrition5k_dataset/imagery/new_realsense_overhead/'+dish+'/rgb.png'

    # rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]   # 1
    rgb_img = Image.open(image_path)

    rgb_img = np.float32(rgb_img) / 255

    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.499, 0.431],
    #                                              std=[0.229, 0.224, 0.225])   # torch.Size([1, 3, 224, 224])
    # rgb_img = torch.tensor(rgb_img)

    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        # todo: check
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        # todo: check
        # must be !
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
    ])

    rgb_img1 = torch.tensor(rgb_img)
    rgb_img1 = torch.transpose(rgb_img1,0,2)
    rgb_img1 = torch.transpose(rgb_img1,1,2)

    input_tensor = img_transforms(rgb_img1)
    input_tensor = input_tensor.view(1,3,256,256)
    #input_tensor = input_tensor.cuda()

    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=False)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    target_category = None # 281

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # 6. 计算cam
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]

    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
    cv2.imwrite(f'/disk/btc010001/New_Nutrition5K/test_picture25/'+dish+'_'+str(DISH.index(t)+1)+'.jpg', visualization)

if __name__ == '__main__':
    img_ori_path = '/disk/btc010001/New_Nutrition5K/nutrition5k_dataset/imagery/new_realsense_overhead'
    img_path = os.listdir('/disk/btc010001/New_Nutrition5K/nutrition5k_dataset/imagery/new_realsense_overhead')
    after_test = []
    with open("/disk/btc010001/New_Nutrition5K/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt","r") as f:  # 打开文件
        data_test = f.read()  # 读取文件
        for i in img_path:
            if (i in data_test):
                after_test.append(i)

    for i in after_test:
        img_file_path = os.path.join(img_ori_path, i,'rgb.png')
        img_array=cv2.imread(img_file_path,cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join('/disk/btc010001/New_Nutrition5K/test_picture25/'+i+'_0'+'.jpg'),img_array)
        for t in DISH:
            models = torch.load('/disk/btc010001/New_Nutrition5K/code/Baseline_anchor/'+t+'25model.pth')
            models = models.cpu()
            cam(models,i,t)