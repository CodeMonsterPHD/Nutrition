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

DISH_ID_INDEX = 0

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


# import  argparse
# def get_argv():
#     args = argparse.ArgumentParser()
#

def calMAE(pre_res,tmp):

    DATA_FIELDNAMES = ["dish_id", tmp]
    # if len(sys.argv) != 4:
    #   raise Exception("Invalid number of arguments\n\n%s" % __doc__)
    ticks = time.strftime("%Y-%m-%d %H:%M", time.localtime())

    groundtruth_csv_path = '../data/vfdl_data/'+tmp+'_test_gt.csv'  # sys.argv[1]
    #predictions_csv_path =   # sys.argv[2]

    groundtruth_data = ReadCsvData(groundtruth_csv_path)
    prediction_data = {}
    pre_data_id = []
    pre_data_volume = []

    each_prediction = {}
    cla = os.listdir('../VFD_dataset/vfdl/test')
    for i in cla :
        sum_one = 0
        cnt = 0
        this_id = os.listdir(os.path.join('../VFD_dataset/vfdl/test', i))
        for id in this_id:
            true_id,trash = id.split('.')

            indx = groundtruth_data['id'].index(true_id)
            true_volume = groundtruth_data['volume'][indx]
            for pre in pre_res:
                if pre[0] == true_id:
                    persent = abs(pre[1] - float(true_volume))/float(true_volume)
                    sum_one = sum_one + persent
                    cnt = cnt+1
        sum_one = sum_one/cnt * 100
        each_prediction[i] = sum_one


    for i in range(0,len(pre_res)):
        pre_data_id.append(pre_res[i][0])
        pre_data_volume.append(pre_res[i][1])
    prediction_data['id'] = pre_data_id
    prediction_data['volume'] = pre_data_volume


    groundtruth_values = {}
    err_values = {}
    output_stats = {}

    for field in DATA_FIELDNAMES[1:]:
        groundtruth_values[field] = []
        err_values[field] = []

    m = 0
    for dish_id in prediction_data['id']:
        k = groundtruth_data['id'].index(dish_id)
        for i in range(1, len(DATA_FIELDNAMES)):
            groundtruth_values[DATA_FIELDNAMES[i]].append(float(groundtruth_data[DATA_FIELDNAMES[i]][k]))
            err_values[DATA_FIELDNAMES[i]].append(abs(
                float(prediction_data[DATA_FIELDNAMES[i]][m])
                - float(groundtruth_data[DATA_FIELDNAMES[i]][k])))
        m = m + 1
            # groundtruth_values[DATA_FIELDNAMES[i]].append(
            #     float(groundtruth_data[dish_id][i]))
            # err_values[DATA_FIELDNAMES[i]].append(abs(
            #     float(prediction_data[dish_id][i])
            #     - float(groundtruth_data[dish_id][i])))

    for field in DATA_FIELDNAMES[1:]:
        output_stats[field + "_MAE"] = statistics.mean(err_values[field])
        output_stats[field + "_MAE_%"] = (100 * statistics.mean(err_values[field]) /
                                          statistics.mean(groundtruth_values[field]))

    # with open(output_path, "w") as f_out:
    #     f_out.write(json.dumps(output_stats))

    return output_stats[tmp+"_MAE"],output_stats[tmp+"_MAE_%"],each_prediction
