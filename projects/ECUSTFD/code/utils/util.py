import json
import torch
from torchvision.datasets import ImageFolder
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms as transforms

from back_best_code.core.function import read_ann_json

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X,image_id,std_mass,section,offset in train_loader:
        X = X.float()
        for d in range(3):
            max = X.max()
            X = X.div(max)
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


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
            img_a = test_transforms(img_a)
            return img_a,image_id,std_mass,section,offset
        return img_a,image_id,std_mass,section,offset

    def __len__(self):
        return len(self.anns)

