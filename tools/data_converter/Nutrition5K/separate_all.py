'''
In Nutrition5K
Separate the all nutrition for the respective evaluation process
We follow the Nutrition5K data division
'''
import csv
from os import path

def creat_csv(food_name):
    path = "/disk/btc010001/nutrition5k/data/data_old/"+food_name+"_test_gt.csv"
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["dish_id", food_name]
        csv_write.writerow(csv_head)

def writer_csv(test,food_name):
    path = "/disk/btc010001/nutrition5k/data/data_old/"+food_name+"_test_gt.csv"
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(test)

def ReadCsvData(filepath):
    if not path.exists(filepath):
        raise Exception("File %s not found" % path)
    parsed_data = {}
    with open(filepath, "r") as f_in:
        file = csv.reader(f_in)
        # filelines = f_in.readlines()
        # for line in filelines:
        #   data_values = line.strip().split(",")
        for i in range(0, 3):
            if i == 1:
                continue
            column = [row[i] for row in file]
            parsed_data[column[0]] = column
            f_in.seek(0)
    return parsed_data

CATEGORY = ["calories","mass","fat","carb","protein"]

del_food = ['dish_1551232973','dish_1551138237','dish_1551389458','dish_1551381990','dish_1560974769']
with open("/disk/btc010001/nutrition5k/data/nutrition5k_dataset/metadata/dish_metadata_overhead.csv", "r",
          encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

# for i in range(0, len(rows['dish_id']) - 1):
#     if rows['dish_id'][i] in del_food:
#         rows['dish_id'].pop(i)
#         rows['calories'].pop(i)
#         rows['mass'].pop(i)
#         rows['fat'].pop(i)
#         rows['carb'].pop(i)
#         rows['protein'].pop(i)

for t in CATEGORY:
    idx = CATEGORY.index(t)
    creat_csv(t)
    # for i in range(0, len(rows['dish_id'])):
    #     data = rows['dish_id'][i], rows[t][i]
    with open(("/disk/btc010001/nutrition5k/data/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt"),"r") as f:  # 打开文件
        datas = f.read()  # 读取文件
        for i in range(0, len(rows)):
            if rows[i][0] in datas:
                writer_csv([rows[i][0],rows[i][idx+1]],t)




