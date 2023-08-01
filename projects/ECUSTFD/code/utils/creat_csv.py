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

#CATEGORY = ["weight(g)"]


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


def creat_csv(food_name):
    path = "/disk/btc010001/ECUSTFD/data/" + food_name + "_test_gt.csv"
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["id", food_name]
        csv_write.writerow(csv_head)

def writer_csv(test, food_name):
    path = "/disk/btc010001/ECUSTFD/data/" + food_name + "_test_gt.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(test)


def main():
    # args = args_parser()
    # segment = args.segment
    segment = 10
    xls_reader = ReadExcel('../../ECUSTFD-resized--master/density.xls')
    all_data = xls_reader.read_excel()
    img_list = os.listdir('../../ECUSTFD-resized--master/JPEGImages')
    after_test = []

    new_ls = []
    for food_name,food_value in all_data.items():
        for id,atri in food_value.items():
            new_ls.append(atri)
    creat_csv("volume(mm^3)")
    for row in new_ls:
        writer_csv([row['id'],row["volume(mm^3)"]], "volume(mm^3)")

if __name__ == '__main__':
    main()