import csv
import json
from os import path
import torch
import numpy as np
import sys

import matplotlib.pyplot as plt
CATEGORY = ["mass","calories","fat","carb","protein"]

def ReadCsvData(filepath):
    if not path.exists(filepath):
        raise Exception("File %s not found" % path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        file = csv.reader(f_in)
        # filelines = f_in.readlines()
        # for line in filelines:
        #   data_values = line.strip().split(",")
        for i in range(0, 6):
            # if i == 1:
            #     continue
            column = [row[i] for row in file]
            parsed_data[column[0]] = column
            f_in.seek(0)
    return parsed_data

def main():
    segment = 10
    del_food = ['dish_1551232973','dish_1551138237','dish_1551389458','dish_1551381990','dish_1560974769','dish_1564159636']  ### remove the error data
    all_data = ReadCsvData('/home/zaiyihu/CodeSpace/Nutrition_code/Nutrition5K/human_ann/metadata/dish_metadata_overhead.csv')
    for i in range(0,len(all_data['dish_id'])-2):
        if all_data['dish_id'][i] in del_food:
            all_data['dish_id'].pop(i)
            all_data['calories'].pop(i)
            all_data['mass'].pop(i)
            all_data['fat'].pop(i)
            all_data['carb'].pop(i)
            all_data['protein'].pop(i)
    train_all_data = []
    test_all_data = []
    for t in CATEGORY:
        for i in range(0,len(all_data['dish_id'])):
            data = all_data['dish_id'][i],all_data[t][i]

        all_data[t] = [float(i) for i in all_data[t][1:]]
        max_tmp = max(all_data[t][1:])
        min_tmp = min(all_data[t][1:])

        seg_size = (max_tmp - min_tmp) / segment
        new_start = min_tmp - seg_size / 2
        new_end = max_tmp + seg_size / 2

        annotations = []
        for u in range(1, len(all_data['dish_id'])-1):
            dict_ann = {}
            dict_ann['dish_id'] = all_data['dish_id'][u]
            dict_ann[t] = all_data[t][u-1]
            annotations.append(dict_ann)

        gt_1 = np.zeros((len(all_data[t]), segment + 1))
        gt_2 = all_data['dish_id'][1:]
        gt_3 = all_data[t][0:]
        gt = {'dish_id': gt_2, t: gt_1, 'true_tmp': gt_3}

        for p in range(0,len(annotations)-1):
            id = annotations[p]['dish_id']
            std_tmp = annotations[p][t]
            k = -1
            for i in range(0, segment + 1):
                if std_tmp > new_start + i * seg_size and std_tmp <= new_start + (i + 1) * seg_size:
                    k = i
                    break
            if id in gt['dish_id']:
                x = gt['dish_id'].index(id)
                gt[t][x][k] = 1
            else:
                print("!")

        new_ann = []
        for u in range(0, len(gt['dish_id'])):
            dict_ann = {}
            dict_ann['dish_id'] = gt['dish_id'][u]
            dict_ann[t] = gt['true_tmp'][u]
            dict_ann['section'] = gt[t][u]
            dict_ann['offset'] = 0

            new_ann.append(dict_ann)
            #writer_csv(tmp)

        train_set,test_set = train_test(new_ann)
        train_all_data = train_all_data + train_set
        test_all_data = test_all_data + test_set

    fig, axes = plt.subplots(nrows=1, ncols=len(CATEGORY), figsize=(15, 5))
    start_train_idx, start_test_idx = 0, 0
    end_train_idx, end_test_idx = int(len(train_all_data) / 5), int(len(test_all_data) / 5)

    for j, category in enumerate(CATEGORY):
        train_data = [data[category] for data in train_all_data[start_train_idx:end_train_idx]]
        test_data = [data[category] for data in test_all_data[start_test_idx:end_test_idx]]

        axes[j].boxplot([train_data, test_data], labels=['Train', 'Test'])
        axes[j].set_title(category)
        axes[j].set_ylabel('Values')

        start_train_idx += int(len(train_all_data) / 5)
        start_test_idx += int(len(test_all_data) / 5)
        end_train_idx += int(len(train_all_data) / 5)
        end_test_idx += int(len(test_all_data) / 5)

    plt.tight_layout()
    plt.show()

        # train_set,train_anchor = get_anchor(segment,train_set,t)
        # test_set,test_anchor = get_anchor(segment,test_set,t)
        # dict_train = {}
        # dict_test = {}
        # dict_train['annotations'] = train_set
        # dict_test['annotations'] = test_set
        # write_json(dict_train,'train',t,segment)
        # write_json(dict_test, 'test', t,segment)
        #
        # print(train_anchor)
        # print(test_anchor)



def train_test(ann):
    ls = list(ann)
    after_train = []
    with open("/home/zaiyihu/CodeSpace/Nutrition_code/Nutrition5K/human_ann/dish_ids/splits/rgb_train_ids.txt", "r") as f:  # 打开文件
        data_train = f.read()  # 读取文件
        for i in ls:
            if(i['dish_id'] in data_train):
                 after_train.append(i)

    ls = list(ann)
    after_test = []
    with open("/home/zaiyihu/CodeSpace/Nutrition_code/Nutrition5K/human_ann/dish_ids/splits/rgb_test_ids.txt","r") as f:  # 打开文件
        data_test = f.read()  # 读取文件
        for i in ls:
            if (i['dish_id'] in data_test):
                after_test.append(i)

    return after_train,after_test

def takeMass(elem):
    return elem['mass']

def takeCal(elem):
    return elem['calories']

def takeFat(elem):
    return elem['fat']

def takeCarb(elem):
    return elem['carb']

def takePro(elem):
    return elem['protein']

def get_anchor(segment,ann,t):
    ls = list(ann)
    after_train = []
    if t == 'mass':
        ls.sort(key=takeMass)
    elif t=='calories':
        ls.sort(key=takeCal)
    elif t== 'fat':
        ls.sort(key=takeFat)
    elif t=='carb':
        ls.sort(key=takeCarb)
    elif t=='protein':
        ls.sort(key=takePro)

    picture_size = int(len(ls)/segment)

    idx = -1
    list_anchor = []

    while idx<len(ls):
        idx = idx+1
        min_ = ls[idx][t]
        idx = idx + picture_size-1
        if len(ls)-idx>=picture_size:
            max_ = ls[idx][t]
        else:
            max_ = ls[-1][t]
            idx = len(ls)
        anchor_set = (min_+max_)/2
        list_anchor.append((min_,max_,anchor_set))
        #print(min_,max_,anchor_set)

    i = -1
    j = 0
    for mass in ls:
        i = i+1
        if i >= picture_size:
            i = 0
            if j != segment-1:
                j = j+1
        section = [0] * segment
        section[j] = 1
        mass['section'] = section
        mass['offset'] = mass[t] - list_anchor[j][2]

    return ls,list_anchor

def write_json(ls,t,phase,segment):
    with open('/home/zaiyihu/CodeSpace/Nutrition_code/Nutrition5K/Baseline_data/'+'change_'+phase+'_'+t+str(segment)+'.json', 'wb') as f1:
        f1.truncate()
        f1.write(json.dumps(ls).encode('utf-8'))
        f1.write('\n'.encode('utf-8'))
        print('save successfully')

if __name__ == '__main__':
    main()