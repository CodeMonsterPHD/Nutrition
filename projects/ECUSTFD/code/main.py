import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from collections import namedtuple
import numpy as np
import json
import random
from torch.backends import cudnn
import shutil
import time
import pandas as pd

from models.Network_ViT import base_patch16_384_gap
from config.args_parse import args_parser
from core.function import train,read_ann_json,evaluate,MylossFunc
from models.network import ResNet, Bottleneck
from dataset.dataset import ECUSTFD
from criterion.compute_eval_statistics import calMAE

#CATEGORY = ["weight(g)","volume(mm^3)"]
CATEGORY = ["weight(g)"]

### set certain seed ###
def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn related setting
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True

r"""
backup codes,the root_dir is where you put your code, the res_dir is where you want to back direct
backup_list is the [name]
"""
def backup_codes(root_dir, res_dir, backup_list):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    for name in backup_list:
        shutil.copytree(os.path.join(root_dir, name), os.path.join(res_dir, name))
    print('codes backup at {}'.format(res_dir))

def main(args,i,ticks):
    fix_randomness(args.seed)
    torch.cuda.empty_cache()
    batch_size = args.batch_size
    output_dim = args.output_dim
    found_lr = args.found_lr
    segment = args.segment
    epoch = args.epoch
    seed = args.seed

    train_img_dir = '../ECUSTFD-resized--master/TrainImage'
    test_img_dir = '../ECUSTFD-resized--master/TestImage'

    train_json_file = '../data/ECUSTFD_'+i+'_train.json'
    test_json_file = '../data/ECUSTFD_'+i+'_test.json'

    train_anchor = [(26.0, 39.8, 32.9), (39.9, 54.5, 47.2), (54.7, 65.6, 60.15), (66.2, 94.2, 80.2), (96.2, 150.5, 123.35), (150.5, 155.1, 152.8), (156.9, 196.0, 176.45), (197.5, 214.5, 206.0), (218.0, 238.0, 228.0), (238.0, 448.0, 343.0)]
    test_anchor = [(26.0, 39.6, 32.8), (39.8, 52.4, 46.099999999999994), (54.5, 70.9, 62.7), (78.3, 92.3, 85.3), (93.2, 104.3, 98.75), (105.8, 154.2, 130.0), (154.2, 177.3, 165.75), (179.3, 219.5, 199.4), (220.5, 255.0, 237.75), (255.0, 448.0, 351.5)]

    #### dataset #####
    train_anns = read_ann_json(train_json_file)
    train_fun = ECUSTFD(train_img_dir, train_anns, phase='train',category=i)
    train_loader = torch.utils.data.DataLoader(train_fun, batch_size=batch_size, shuffle=True,num_workers = 8,pin_memory = True)

    test_anns = read_ann_json(test_json_file)
    test_fun = ECUSTFD(test_img_dir, test_anns, phase='test',category=i)
    test_loader = torch.utils.data.DataLoader(test_fun, batch_size=batch_size, shuffle=False,num_workers = 8,pin_memory = True)
    #### dataset #####


    ##### network #####
    if args.model == 'ResNet':
        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet50_config = ResNetConfig(block=Bottleneck,
                                    n_blocks=[3, 4, 6, 3],
                                    channels=[64, 128, 256, 512])

        resnet50 = models.resnet50(pretrained=True)
        model = ResNet(resnet50_config, output_dim)
        # 读取参数
        pretrained_dict = resnet50.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)

        # 加载我们真正需要的state_dict
        model.load_state_dict(model_dict,True)

        learnable_pam = MylossFunc()
        #### optimizer #####
        params = [
        {'params': model.conv1.parameters(), 'lr': found_lr / 10},
        {'params': model.bn1.parameters(), 'lr': found_lr / 10},
        {'params': model.layer1.parameters(), 'lr': found_lr / 8},
        {'params': model.layer2.parameters(), 'lr': found_lr / 6},
        {'params': model.layer3.parameters(), 'lr': found_lr / 4},
        {'params': model.layer4.parameters(), 'lr': found_lr / 2},
        {'params': model.fc_self.parameters(), 'lr': found_lr / 2},
        {'params': model.fc1.parameters(), 'lr': found_lr / 2},
        {'params': model.fc2.parameters(), 'lr': found_lr / 2},
        {'params': model.lastlayer.parameters()},
        {'params': learnable_pam.parameters()}
    ]

    #optimizer = optim.RMSprop(params, lr=found_lr,momentum= 0.9,weight_decay= 0.9,eps=1.0)
    #optimizer = optim.Adam(params, lr=found_lr,weight_decay=0.9,eps=1.0)
        optimizer = optim.Adam(params, lr=found_lr)

    #### optimizer #####
    elif args.model == 'ViT':
        model = base_patch16_384_gap()
        optimizer = torch.optim.Adam(
            [  #
                {'params': model.parameters(), 'lr': args.found_lr},
            ], lr=args.found_lr)
    model = model.cuda()
    learnable_pam = MylossFunc()
    #### network #####

    food_MAE_pre = 100
    for iepoch in range(epoch):
        loss_sec,loss_off = train(args,train_loader, model,learnable_pam, optimizer,batch_size,segment,train_anchor)
        print('epoch ' + str(iepoch) + '\n'+ f'\tTrain Section_Loss: {loss_sec:.3f}'+'\n'+f'\tTrain Offset_Loss: {loss_off:.3f}\n')
        pre_res = evaluate(test_loader,model,learnable_pam,batch_size,segment,test_anchor,i)
        #predcts_test = evaluate(model,test_loader,batch_size)
        final_err_values = calMAE(pre_res,i)
        print(final_err_values)
        tmp_error = 0
        cnt = 0
        for k,v in final_err_values.items():
            cnt = cnt + 1
            tmp_error = tmp_error + v

        tmp_error = tmp_error/cnt
        if tmp_error<food_MAE_pre:
            food_MAE_pre = tmp_error
            # with open("/disk/btc010001/ECUSTFD/result/" + ticks + "ECUSTFD_prediction.txt", "a") as f:
            #     f.write(final_err_values)
            final_value = list(final_err_values.items())
            df = pd.DataFrame(final_value)
            df.to_excel('../result/'+ticks+'seed'+str(seed)+ 'volume(mm^3)' +'.xlsx', index=True)
            torch.save(model,i+ticks+'CE+LLS'+'model.pth')
            print(f'\t '+'volume(mm^3)'+f'_MAE:{food_MAE_pre:.3f}\n')

    print(f'\t ' + i + f'_MAE:{food_MAE_pre:.3f}\n')

if __name__ == '__main__':
    args= args_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_DEVICE

    ticks = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    print(ticks)
    # root_dir = '../code/..'
    # res_dir = '../code/../ckpt/ecustfd/'+ ticks +'volume_imp_code'
    # #backup_codes(root_dir,res_dir,['volume_code'])

    for i in CATEGORY:
        main(args,i,ticks)

    tick2 = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    print(tick2)
