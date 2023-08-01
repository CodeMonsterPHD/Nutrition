import csv
import json
from os import path
import torch
import numpy as np
import sys
import os
import pandas as pd
import xlrd
import math

# def writer_csv(test):
#     path = "../data_old/tmp_overhead.csv"
#     with open(path, 'a+') as f:
#         csv_write = csv.writer(f)
#         csv_write.writerow(test)

CATEGORY = ["weight(g)","volume(mm^3)"]


class ReadExcel:
    def __init__(self, path):
        self.path = path

    def read_excel(self):
        """
        遍历excel所有sheet，并以字典返回
        :param row:
        :return:
        """
        with xlrd.open_workbook(self.path, 'rb') as book:
            sheets = book.sheet_names()  # 找到所有sheets
            data_dict = {}
            for sheet in sheets:
                table = book.sheet_by_name(sheet)  # 找到要操作的sheet

                # 获取sheet所有列数
                col_num = table.ncols
                #  读取第一行的值，作为每个sheet返回字典的key
                keys = table.row_values(0)

                # # 读取指定行，作为每个sheet返回字典的value
                # values = table.row_values(row)
                row_num = table.nrows
                sheet_dict = {}
                for row in range(1,row_num):
                    values = table.row_values(row)
                    # 遍历所有列，并以字典接收,其中第一行作为字典的key，其他行作为字典的value
                    row_dict = {}
                    for col in range(col_num):
                        row_dict[keys[col]] = values[col]

                    # 遍历所有sheet，并以字典接收返回，其中sheet名称作为字典的key，每个sheet的数据作为字典的value
                    sheet_dict[values[0]] = row_dict
                data_dict[sheet] = sheet_dict
        return data_dict


def main():
    # args = args_parser()
    # segment = args.segment
    segment = 10
    xls_reader = ReadExcel('../../ECUSTFD-resized--master/density.xls')
    all_data = xls_reader.read_excel()

    for t in CATEGORY:
        train_set,train_anchor = get_anchor(segment,all_data,t,'train')
        test_set, test_anchor = get_anchor(segment,all_data,t,'test')
        #test_set,test_anchor = get_anchor(segment,test_set,t)
        dict_train = {}
        dict_test = {}
        dict_train['annotations'] = train_set
        dict_test['annotations'] = test_set
        write_json(dict_train,'train',t)
        write_json(dict_test, 'test', t)
        print(train_anchor)
        print(test_anchor)

        print(1)


def takeWeight(elem):
    return elem['weight(g)']

def takeVolume(elem):
    return elem['volume(mm^3)']

def get_anchor(segment,ann,t,phase):
    after_train = []
    img_list =  os.listdir('../../ECUSTFD-resized--master/JPEGImages')
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

    train_picture_size = math.ceil(len(after_train)/(segment))
    test_picture_size = math.ceil((len(after_test)/(segment)))

    if phase == 'train':
        pass
    else:
        train_picture_size = test_picture_size
        after_train = after_test

    new_ls = []
    for food_name,food_value in ann.items():
        for id,atri in food_value.items():
            new_ls.append(atri)
    if t == 'weight(g)':
        new_ls.sort(key = takeWeight)
    else:
        new_ls.sort(key=takeVolume)
    ls = new_ls
    idx = 0
    u = 0
    train_list_anchor = []
    while idx<len(ls) and u<len(after_train):
        u = u + train_picture_size
        min_ = ls[idx][t]
        cnt = 0
        for j in ls:
            for i in after_train:
                if 'S' in i:
                    img_name,tmp = i.split('S')
                elif 'T' in i:
                    img_name,tmp =i.split('T')
                if img_name == j['id']:
                    cnt = cnt + 1
                if cnt == u:
                    max_ = j[t]
                    break
            if cnt == u:
                max_ = j[t]
                break
        idx = ls.index(j)
        idx = idx + 1
        if len(train_list_anchor) == segment-1:
            max_ = ls[-1][t]
        anchor_set = (min_+max_)/2
        train_list_anchor.append((min_,max_,anchor_set))
        #print(min_,max_,anchor_set)

    food_ls = []
    for food in ls:
        for i in after_train:
            food_dict = {}
            img_name, tmp = i.split('.')
            if 'S' in i:
                part_img_name, tmp = i.split('S')
            elif 'T' in i:
                part_img_name, tmp = i.split('T')
            if part_img_name == food['id']:
                section = []
                for idx in range(0,segment):
                    if food[t]>=train_list_anchor[idx][0] and food[t]<=train_list_anchor[idx][1]:
                        section.append(1.)
                        offset = food[t] - train_list_anchor[idx][2]
                    else:
                        section.append(0.)
                food_dict['id'] = img_name
                food_dict['type'] = food['type']
                food_dict[t] = food[t]
                food_dict['section'] = section
                food_dict['offset'] = offset
                food_ls.append(food_dict)

    return food_ls,train_list_anchor

def write_json(ls,t,phase):
    with open('../data/ECUSTFD_'+phase+'_'+t+'.json', 'wb') as f1:
        f1.truncate()
        f1.write(json.dumps(ls).encode('utf-8'))
        f1.write('\n'.encode('utf-8'))
        print('save successfully')

if __name__ == '__main__':
    main()