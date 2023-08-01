import csv
import os
import pandas as pd
import xlrd

#CATEGORY = ["weight(g)"]

class ReadExcel:
    def __init__(self, path):
        self.path = path

    def read_excel(self):
        """
        :param row:
        :return:
        """
        with xlrd.open_workbook(self.path, 'rb') as book:
            sheets = book.sheet_names()  #
            data_dict = {}
            for sheet in sheets:
                table = book.sheet_by_name(sheet)  #
                col_num = table.ncols
                keys = table.row_values(0)
                # values = table.row_values(row)
                row_num = table.nrows
                sheet_dict = {}
                for row in range(1,row_num):
                    values = table.row_values(row)
                    #
                    row_dict = {}
                    for col in range(col_num):
                        row_dict[keys[col]] = values[col]

                    #
                    sheet_dict[values[0]] = row_dict
                data_dict[sheet] = sheet_dict
        return data_dict


def creat_csv(food_name):
    path = "./lib/evaluation/data_gt/" + food_name + "_test_gt.csv"
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["id", food_name]
        csv_write.writerow(csv_head)

def writer_csv(test, food_name):
    path = "./lib/evaluation/data_gt/" + food_name + "_test_gt.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(test)


def main():
    # args = args_parser()
    # segment = args.segment
    xls_reader = ReadExcel('./lib/dataset/density.xls')
    all_data = xls_reader.read_excel()

    new_ls = []
    for food_name,food_value in all_data.items():
        for id,atri in food_value.items():
            new_ls.append(atri)
    creat_csv("volume(mm^3)")
    for row in new_ls:
        writer_csv([row['id'],row["volume(mm^3)"]], "volume(mm^3)")

if __name__ == '__main__':
    main()