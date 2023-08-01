import csv
import json
from os import path
import torch
import numpy as np
import os
import sys
import math

from back_best_code.config.args_parse import args_parser


# def writer_csv(test):
#     path = "../data_old/tmp_overhead.csv"
#     with open(path, 'a+') as f:
#         csv_write = csv.writer(f)
#         csv_write.writerow(test)

CATEGORY = ["volume"]

def ReadCsvData(filepath):
    if not path.exists(filepath):
        raise Exception("File %s not found" % path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        file = csv.reader(f_in)
        # filelines = f_in.readlines()
        # for line in filelines:
        #   data_values = line.strip().split(",")
        for i in range(0, 1):
            # if i == 1:
            #     continue
            column = [row[i] for row in file]
            parsed_data[column[0]] = column
            f_in.seek(0)
    return parsed_data

def main():
    segment = 10

    all_data = ReadCsvData('../../VFD_dataset/vfdl/pittv8.csv')


    for t in CATEGORY:
        train_set,train_anchor = get_anchor(segment,all_data,t,'train')
        test_set, test_anchor = get_anchor(segment,all_data,t,'test')
        dict_train = {}
        dict_test = {}
        dict_train['annotations'] = train_set
        dict_test['annotations'] = test_set
        write_json(dict_train,'train',t,segment)
        write_json(dict_test, 'test', t,segment)
        creat_test(all_data,t)
        print(train_anchor)
        print(test_anchor)


def takeVolume(elem):
    return elem['volume']

def get_anchor(segment,ann,t,phase):
    after_train = []
    train_img_list = os.listdir('../../VFD_dataset/vfdl/train_resize')
    test_img_list = os.listdir('../../VFD_dataset/vfdl/test_resize')
    for i in range(0,len(ann[t])):
        tmp = str(i+1) + '.jpg'
        if tmp in train_img_list:
            after_train.append(i+1)

    after_test = []
    for i in range(0,len(ann[t])):
        tmp = str(i+1) + '.jpg'
        if tmp in test_img_list:
            after_test.append(i+1)

    train_picture_size = math.ceil(len(after_train)/(segment))
    test_picture_size = math.ceil((len(after_test)/(segment)))

    new_ann = {}
    new_list = []
    for i in range(0,len(ann[t])):
        if i+1 in after_train or i+1 in after_test:
            new_ann['id'] = i+1
            new_ann['volume'] = float(ann[t][i+1])
            new_list.append(new_ann)
            new_ann = {}

    train_ls = []
    test_ls = []
    new_list.sort(key=takeVolume)
    for i in new_list:
        if i['id'] in after_train:
            train_ls.append(i)
        else:
            test_ls.append(i)

    if phase == 'train':
        pass
    else:
        train_picture_size = test_picture_size
        after_train = after_test
        train_ls = test_ls

    idx = 0

    train_list_anchor = []
    while idx<len(train_ls):
        min_ = train_ls[idx][t]
        cnt = 0
        idx = idx + train_picture_size-1
        if len(train_ls)-idx>=train_picture_size:
            max_ = train_ls[idx][t]
        else:
            max_ = train_ls[-1][t]
            idx = len(train_ls)
        anchor_set = (min_+max_)/2
        train_list_anchor.append((min_,max_,anchor_set))

        #print(min_,max_,anchor_set)

    food_ls = []
    for food in train_ls:
        food_dict = {}
        section = []
        for idx in range(0,segment):
            if food[t]>=train_list_anchor[idx][0] and food[t]<=train_list_anchor[idx][1]:
                section.append(1.)
                offset = food[t] - train_list_anchor[idx][2]
            else:
                section.append(0.)
        food_dict['id'] = food['id']
        food_dict[t] = food[t]
        food_dict['section'] = section
        food_dict['offset'] = offset
        food_ls.append(food_dict)

    return food_ls,train_list_anchor

def write_json(ls,t,phase,segment):
    with open('../data/vfdl_data/'+'VFD_'+phase+'_'+t +'_seg'+str(segment)+'.json', 'wb') as f1:
        f1.truncate()
        f1.write(json.dumps(ls).encode('utf-8'))
        f1.write('\n'.encode('utf-8'))
        print('save successfully')

def creat_test(ann,t):
    test_img_list = os.listdir('../../VFD_dataset/vfdl/test_resize')
    after_test = []
    for i in range(0,len(ann[t])):
        tmp = str(i+1) + '.jpg'
        if tmp in test_img_list:
            after_test.append(i+1)

    new_ann = {}
    creat_csv("volume")
    for i in range(0,len(ann[t])):
        if i+1 in after_test:
            new_ann['id'] = i+1
            new_ann['volume'] = float(ann[t][i+1])
            writer_csv([new_ann['id'], new_ann['volume']], "volume")
            new_ann = {}

def creat_csv(food_name):
    path = "../data/vfdl_data/" + food_name + "_test_gt.csv"
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["id", food_name]
        csv_write.writerow(csv_head)

def writer_csv(test, food_name):
    path = "../data/vfdl_data/" + food_name + "_test_gt.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(test)

if __name__ == '__main__':
    main()