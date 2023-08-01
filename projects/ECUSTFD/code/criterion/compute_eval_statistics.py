# r"""Script to compute statistics on nutrition predictions.
#
# This script takes in a csv of nutrition predictions and computes absolute and
# percentage mean average error values comparable to the metrics used to eval
# models in the Nutrition5k paper. The input csv file of nutrition predictions
# should be in the form of:
# dish_id, calories, mass, carbs, protein
# And the groundtruth values will be pulled from the metadata csv file provided
# in the Nutrition5k dataset release where the first 5 fields are also:
# dish_id, calories, mass, carbs, protein
#
# Example Usage:
# python compute_statistics.py path/to/groundtruth.csv path/to/predictions.csv \
# path/to/output_statistics.json
# """
import csv
import json
import os
from os import path
import statistics
import sys
import time
import xlrd

DISH_ID_INDEX = 0
REAL_DENSITY = {}
CATEGORY = ['apple','banana','bread','bun','doughnut','egg','fireddoughtwist','grape','lemon','litchi','mango','mix','mooncake','orange','peach','pear','plum','qiwi','sachima','tomato']
DENSITY = [0.78,0.91,0.18,0.34,0.31,1.03,0.58,0.97,0.96,1.00,1.07,1,0.96,0.90,0.96,1.02,1.01,0.97,0.22,0.98]

def ReadCsvData(filepath):
    # if not path.exists(filepath):
    #     raise Exception("File %s not found" % path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        file = csv.reader(f_in)
        # filelines = f_in.readlines()
        # for line in filelines:
        #   data_values = line.strip().split(",")
        for i in range(0, 2):
            column = [row[i] for row in file]
            parsed_data[column[DISH_ID_INDEX]] = column
            f_in.seek(0)
    return parsed_data

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


def calMAE(pre_res,tmp):

    DATA_FIELDNAMES = ["id", tmp]
    # if len(sys.argv) != 4:
    #   raise Exception("Invalid number of arguments\n\n%s" % __doc__)
    ticks = time.strftime("%Y-%m-%d %H:%M", time.localtime())

    groundtruth_csv_path = '../data/'+tmp+'_test_gt.csv'  # sys.argv[1]
    #predictions_csv_path = pre_res  # sys.argv[2]
    #output_path = ticks + 'output_statistics.json'  # sys.argv[3]

    groundtruth_data = ReadCsvData(groundtruth_csv_path)
    #prediction_data = ReadCsvData(predictions_csv_path)

    groundtruth_values = {}

    output_stats = {}

    for field in DATA_FIELDNAMES[1:]:
        groundtruth_values[field] = []

    err_ME = []


    xls_reader = ReadExcel('../ECUSTFD-resized--master/density.xls')
    all_data = xls_reader.read_excel()
    new_ls = []
    for food_name,food_value in all_data.items():
        for id,atri in food_value.items():
            new_ls.append(atri)
    for i in range(0,len(new_ls)):
        new_ls[i]['density'] = new_ls[i]['weight(g)']/new_ls[i]['volume(mm^3)']

    for pred in pre_res:
        food_name = pred[0]
        food_value = pred[1]
        if 'S' in food_name:
            food_n,temp = food_name.split('S')
        elif 'T' in food_name:
            food_n, temp = food_name.split('T')
        idx = groundtruth_data['id'].index(food_n)
        true_value = groundtruth_data[tmp][idx]
        ME = (float(food_value)-float(true_value))/float(true_value)
        err_ME.append([food_n,ME])

    err_values = {}
    all_err_values = {}
    for i in range(0, len(err_ME)):
        food_name = err_ME[i][0]
        true_name = ''.join([x for x in food_name if x.isalpha()])
        err_values[ food_name] = 100
        all_err_values[true_name] = [0,0]

    record_best = {}
    for i in range(0,len(err_ME)-1):
        if err_ME[i][0] == err_ME[i+1][0]:
            food_name = err_ME[i][0]
            min_value = min(abs(err_ME[i][1]),abs(err_ME[i+1][1]))
            if min_value == abs(err_ME[i][1]):
                idx = i
            else:
                idx = i + 1
            if (min_value<err_values[ food_name]):
                err_values[food_name] = min_value
                record_best[food_name] = idx

    volume_dict = {}
    for k,idx in record_best.items():
        food_name = pre_res[idx][0]
        food_value = pre_res[idx][1]
        if 'S' in food_name:
            food_n,temp = food_name.split('S')
        elif 'T' in food_name:
            food_n, temp = food_name.split('T')
        for food in new_ls:
            if food['id'] == food_n:
                pre_volume = food_value/food['density']
                volume_ME = abs(pre_volume-food['volume(mm^3)'])/food['volume(mm^3)']
                volume_dict[food_n] = volume_ME

    # for i in range(0,len(CATEGORY)):
    #     REAL_DENSITY[CATEGORY[i]] = DENSITY[i]
    #
    # volume_dict = {}
    # for k,idx in record_best.items():
    #     food_name = pre_res[idx][0]
    #     food_value = pre_res[idx][1]
    #     if 'S' in food_name:
    #         food_n,temp = food_name.split('S')
    #     elif 'T' in food_name:
    #         food_n, temp = food_name.split('T')
    #     real_food_name = ''.join([x for x in food_n if x.isalpha()])
    #     for real_name,density in REAL_DENSITY.items():
    #         if real_name == real_food_name:
    #             pre_volume = food_value/density
    #             for food in new_ls:
    #                 if food['id'] == food_n:
    #                     volume_ME = abs(pre_volume - food['volume(mm^3)']) / food['volume(mm^3)']
    #                     volume_dict[food_n] = volume_ME

    for k,v in volume_dict.items():
        real_food_name = ''.join([x for x in k if x.isalpha()])
        all_err_values[real_food_name][0] = all_err_values[real_food_name][0] + v
        all_err_values[real_food_name][1] = all_err_values[real_food_name][1] + 1

    final_err_values = {}
    for k,v in all_err_values.items():
        true_value = v[0]/v[1]
        final_err_values[k] = true_value*100

    return final_err_values


