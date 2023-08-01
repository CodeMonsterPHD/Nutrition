import csv
import json
import os
import pandas as pd
import xlrd
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from collections import Counter
import random

import random
random.seed(42)

def random_resampling(train_set):
    # 统计各个 section 的数量
    labels = [tuple(sample['section']) for sample in train_set]
    label_counts = Counter(labels)

    counts = label_counts.values()

    section_counts = list(counts)
    # section_counts[8] 在训练集中数量为0
    section_counts.append(24)
    section_counts[8] = 0
    # 找到占比最高的 section 的数量
    max_count = max(section_counts)

    # 随机重采样其他 section，使数量与占比最高的 section 一致
    resampled_train_set = []
    visited = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for data in train_set:
        section = data['section']
        max_section = max(section)
        max_section_index = section.index(max_section)

        # 判断是否需要重采样
        if section_counts[max_section_index] < max_count and visited[max_section_index] == 0:
            # 计算需要重采样的数量
            resample_count = max_count - section_counts[max_section_index]
            if tuple(section)[5]==1:
                resampled_train_set.append(data)
                continue
            if tuple(section)[6]==1 or tuple(section)[7]==1 or tuple(section)[9]==1:
                resample_count = section_counts[5] - section_counts[max_section_index]
            # 找到具有相同 section 的数据集合
            same_section_data = [sample for sample in train_set if tuple(sample['section']) == tuple(section)]

            resample_data = random.choices(same_section_data, k=resample_count)
            resampled_train_set.extend(resample_data)
            visited[max_section_index] = 1

        # 添加原始数据
        resampled_train_set.append(data)

    return resampled_train_set


# def writer_csv(test):
#     path = "../data_old/tmp_overhead.csv"
#     with open(path, 'a+') as f:
#         csv_write = csv.writer(f)
#         csv_write.writerow(test)

CATEGORY = ["weight(g)", "volume(mm^3)"]


class ReadExcel:
    def __init__(self, path):
        self.path = path

    def read_excel(self):
        """
        :param row:
        :return:
        """
        with xlrd.open_workbook(self.path, 'rb') as book:
            sheets = book.sheet_names()
            data_dict = {}
            for sheet in sheets:
                table = book.sheet_by_name(sheet)
                col_num = table.ncols
                keys = table.row_values(0)
                # values = table.row_values(row)
                row_num = table.nrows
                sheet_dict = {}
                for row in range(1, row_num):
                    values = table.row_values(row)
                    row_dict = {}
                    for col in range(col_num):
                        row_dict[keys[col]] = values[col]
                    sheet_dict[values[0]] = row_dict
                data_dict[sheet] = sheet_dict
        return data_dict


def main():
    # args = args_parser()
    # segment = args.segment
    segment = 10
    xls_reader = ReadExcel('../lib/dataset/density.xls')
    all_data = xls_reader.read_excel()

    for t in CATEGORY:
        train_set, train_anchor = get_anchor(segment, all_data, t, 'train')
        test_set, test_anchor = get_anchor(segment, all_data, t, 'test')
        # plot_type_distribution(train_set)
        resampled_train_set = random_resampling(train_set)
        # plot_type_distribution(resampled_train_set)
        dict_train = {}
        dict_test = {}
        dict_train['annotations'] = resampled_train_set
        dict_test['annotations'] = test_set

        write_json(dict_train, 'train', t)
        write_json(dict_test, 'test', t)
        print(train_anchor)
        print(test_anchor)

    #
    # for t in CATEGORY:
    #     train_set, train_anchor = get_anchor(segment, all_data, t, 'train')
    #     test_set, test_anchor = get_anchor(segment, all_data, t, 'test')
    #     # plot_type_distribution(train_set)
    #
    #     dict_train = {'annotations': train_set}
    #     dict_test = {'annotations': test_set}
    #     synthetic_samples = []
    #     # plot_type_distribution(dataset=train_set)
    #     # Identify classes with fewer samples
    #     class_samples = train_set
    #     class_types = np.array([sample['type'] for sample in class_samples]).reshape(-1, 1)
    #     class_labels = np.array([sample['section'] for sample in class_samples]).astype('str')
    #     class_feat = np.array([sample[t] for sample in class_samples]).astype('str')
    #     class_offset = np.array([sample['offset'] for sample in class_samples]).astype('str')
    #     # Oversample minority class using SMOTE
    #
    #     smote = SMOTE(k_neighbors=2,sampling_strategy='auto',random_state=42)
    #     lb = LabelBinarizer()
    #     class_types_transform = lb.fit_transform(class_types)
    #     synthetic_labels, synthetic_types_transform, = smote.fit_resample(class_labels,class_types_transform)
    #     synthetic_types = lb.inverse_transform(synthetic_types_transform)
    #     synthetic_labels, synthetic_offset, = smote.fit_resample(class_labels, class_offset)
    #     synthetic_labels, synthetic_feat, = smote.fit_resample(class_labels, class_feat)
    #     # Create synthetic samples with corresponding dish_id and weight/volume
    #     type_count = 0
    #     for i in range(len(synthetic_types)):
    #         if synthetic_types[i]!=synthetic_types[i-1]:
    #             type_count = 0
    #         synthetic_sample = {
    #             'id':   'synthetic_' + synthetic_types[i] + '_' + str(type_count),
    #             'type': synthetic_types[i],
    #              t: synthetic_feat[i],  # or 'volume(mm^3)' depending on the required format
    #             'section': synthetic_labels[i],
    #             'offset': synthetic_offset[i]
    #         }
    #         type_count += 1
    #         synthetic_samples.append(synthetic_sample)
    #
    #     # Combine original and synthetic samples
    #     train_set_extended = train_set + synthetic_samples
    #
    #     # Optionally, shuffle the data
    #     # np.random.shuffle(train_set_extended)
    #     # plot_type_distribution(synthetic_samples)
    #     write_json(dict_train, 'train', t)
    #     write_json(dict_test, 'test', t)
    #
    #     print(train_anchor)
    #     print(test_anchor)


def takeWeight(elem):
    return elem['weight(g)']


def takeVolume(elem):
    return elem['volume(mm^3)']


def get_anchor(segment, ann, t, phase):
    after_train = []
    img_list = os.listdir('/data1/zaiyihu/Datasets/Nutrition_data/ECUSTFD/data/JPEGImages')
    with open("../human_ann/ImageSets/Main/trainval.txt", "r") as f:  #
        data_train = f.read()
        for i in img_list:
            u, v = i.split('.')
            if (u in data_train):
                after_train.append(i)
    after_test = []
    with open("../human_ann/ImageSets/Main/test.txt", "r") as f:  #
        data_test = f.read()  #
        for i in img_list:
            u, v = i.split('.')
            if (u in data_test):
                after_test.append(i)

    train_picture_size = math.ceil(len(after_train) / (segment))
    test_picture_size = math.ceil((len(after_test) / (segment)))

    if phase == 'train':
        pass
    else:
        train_picture_size = test_picture_size
        after_train = after_test

    new_ls = []
    for food_name, food_value in ann.items():
        for id, atri in food_value.items():
            new_ls.append(atri)
    if t == 'weight(g)':
        new_ls.sort(key=takeWeight)
    else:
        new_ls.sort(key=takeVolume)
    ls = new_ls
    idx = 0
    u = 0
    train_list_anchor = []
    train_list_anchor = [(26.0, 68.2, 47.1), (68.2, 110.4, 89.3), (110.4, 152.6, 131.5), (152.6, 194.8, 173.7),
                         (194.8, 237, 215.9), (237, 279.2, 258.1), (279.2, 321.4, 300.3), (321.4, 363.6, 342.5),
                         (363.6, 405.8, 384.8), (405.8, 448.0, 426.9)]
    # test_anchor = [(26.0, 68.2, 47.1), (68.2, 110.4, 89.3), (110.4, 152.6, 131.5), (152.6, 194.8, 173.7),
    #                 (194.8, 237, 215.9), (237, 279.2, 258.1), (279.2, 321.4, 300.3), (321.4, 363.6, 342.5),
    #                 (363.6, 405.8, 384.8), (405.8, 448.0, 426.9)]
    # while idx<len(ls) and u<len(after_train):
    #     u = u + train_picture_size
    #     min_ = ls[idx][t]
    #     cnt = 0
    #     for j in ls:
    #         for i in after_train:
    #             if 'S' in i:
    #                 img_name,tmp = i.split('S')
    #             elif 'T' in i:
    #                 img_name,tmp =i.split('T')
    #             if img_name == j['id']:
    #                 cnt = cnt + 1
    #             if cnt == u:
    #                 max_ = j[t]
    #                 break
    #         if cnt == u:
    #             max_ = j[t]
    #             break
    #     idx = ls.index(j)
    #     idx = idx + 1
    #     if len(train_list_anchor) == segment-1:
    #         max_ = ls[-1][t]
    #     anchor_set = (min_+max_)/2
    #     train_list_anchor.append((min_,max_,anchor_set))
    # print(min_,max_,anchor_set)

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
                flag = 0
                for idx in range(0, segment):
                    if flag == 1:
                        section.append(0.)

                    elif food[t] >= train_list_anchor[idx][0] and food[t] <= train_list_anchor[idx][1]:
                        section.append(1.)
                        flag = 1
                        offset = food[t] - train_list_anchor[idx][2]
                    else:
                        section.append(0.)
                food_dict['id'] = img_name
                food_dict['type'] = food['type']
                food_dict[t] = food[t]
                food_dict['section'] = section
                food_dict['offset'] = offset
                food_ls.append(food_dict)

    return food_ls, train_list_anchor


def write_json(ls, t, phase):
    with open('../lib/dataset/ecu_Bin_number10/ECUSTFD_equal-interval_oversampling679_' + phase + '_' + t + '.json', 'wb') as f1:
        f1.truncate()
        f1.write(json.dumps(ls).encode('utf-8'))
        f1.write('\n'.encode('utf-8'))
        print('save successfully')


import matplotlib.pyplot as plt
from collections import Counter


def section_conversion(lst):
    result = []
    for sub_lst in lst:
        for i, num in enumerate(sub_lst):
            if num == 1:
                result.append(i)
                break
    return result


def plot_type_distribution(dataset):
    labels = [tuple(sample['section']) for sample in dataset]
    label_counts = Counter(labels)

    types = label_counts.keys()
    counts = label_counts.values()
    # 计算标准差和偏度
    std_dev = np.std(list(counts))
    skewness = skew(list(counts))

    # 打印统计结果
    print("Type Distribution:")
    print(counts)
    print("Standard Deviation:", std_dev)
    print("Skewness:", skewness)
    # Plot the type distribution
    num_types = len(types)
    x = range(num_types)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)

    rects = ax.bar(x, counts)

    ax.set_xlabel('Type')
    ax.set_ylabel('Count')
    ax.set_title('ECU Train Set Distribution')
    ax.set_xticks(x)

    ax.set_xticklabels(section_conversion(types))
    # plt.xticks(rotation=30, ha='right')

    plt.show()


from scipy.stats import skew

# Call the function to plot the label distribution


if __name__ == '__main__':
    main()
