import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms
from os import path
import json
import torch

class ECUSTFD(data.Dataset):
    def __init__(self, dir, anns, phase,category):
        self.dir = dir
        self.anns = anns
        # phase: train, val, test
        assert phase in ('train',  'test')
        self.phase = phase
        assert category in ("weight(g)","volume(mm^3)")
        self.category = category

    def __getitem__(self, item):
        pretrained_size = [256, 256]
        pretrained_means = [0.485, 0.499, 0.431]
        pretrained_stds = [0.229, 0.224, 0.225]
        img_transforms = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
            # must be !
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
        ])
        if self.phase == 'train':
            data = self.anns[item]
            image_id = data['id']
            img_type = data['type']
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img_path = os.path.join(self.dir, image_id+".JPG")
            img_a = Image.open(img_path)
            img_a = img_a.resize((384,384),Image.ANTIALIAS)
            img_a = img_transforms(img_a)
        else:
            data = self.anns[item]
            image_id = data['id']
            img_type = data['type']
            ###there only mass####
            std_mass = data[self.category]
            section = np.array(data['section'])
            offset = data['offset']

            ### there image from release_head.rgb.png####
            img_path = os.path.join(self.dir, image_id+ ".JPG")
            img_a = Image.open(img_path)
            img_a = img_a.resize((384,384),Image.ANTIALIAS)
            img_a = test_transforms(img_a)
            return img_a,image_id,std_mass,section,offset
        return img_a,image_id,std_mass,section,offset

    def __len__(self):
        return len(self.anns)

####There this function not be used #####
def ReadCsvData(filepath):
    if not path.exists(filepath):
        raise Exception("File %s not found" % path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        file = csv.reader(f_in)
        # filelines = f_in.readlines()
        # for line in filelines:
        #   data_values = line.strip().split(",")
        for i in range(0, 2):
            column = [row[i] for row in file]
            parsed_data[column[0]] = column
            f_in.seek(0)
    return parsed_data
####There this function not be used #####


if __name__ == '__main__':
    import json
    import torch
    import csv
    train_img_dir = '/disk/btc010001/ECUSTFD/ECUSTFD-resized--master/TrainImage'
    file = '/data/ECUSTFD_weight(g)_train.json'

    with open(file, 'r') as f:
        datas = json.load(f)
    annotations = datas['annotations']

    train_fun = ECUSTFD(train_img_dir, annotations, phase='train',category='weight(g)')
    train_loader = torch.utils.data.DataLoader(train_fun, batch_size = 4, shuffle=True)

    for img1,image_id,std_mass,section,offset in train_loader:
        # if (img1 is None or img2 is  None or img3 is None or img4 is None):
        #     pass
        # else:
        img1 = img1.cuda()
        print(type(img1), img1.shape)
        print(image_id,std_mass,section,offset)
        # print(image_id)
        # print(std_cal)
        # print(std_mass)
        # print(std_fat)
        # print(std_carb)
        # print(std_pro)


