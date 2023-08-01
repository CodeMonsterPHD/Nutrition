# Nutrition: Coarse-to-Fine Nutrition Prediction 

This repository is the official implementation of our method. In this work, we propose the coarse-to-fine paradigm combined with structure loss to predict nutrition. 

![](.\\sources\\Framework.png)


## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
```

Before running the code, please activate this conda environment.

## Data Preparation

Download food images of Nutrition5K dataset and our best model from [Baidu Netdisk]( https://pan.baidu.com/s/1pf_A0F8rFZzTMi-1Nsp2zg).

acess code：gw8x

Please ensure the data structure is as below

~~~~
Nutrition5K,ECUSTFD,VFD
├── data
   └── Nutrition5K
       └── imagery
           └── new_realsense_overhead           
               ├── dish_1556572657
                   └── rgb.png
               ├── dish_1556573514                  
                   └── rgb.png
               └── ...
   └── ECUSTFD
       ├── TrainImage           
           ├── apple001S(1).JPG
           ├── apple001S(2).JPG
           └── ...
       └── TestImage
           ├── apple015S(1).JPG
           ├── apple015S(2).JPG
           └── ...
   └── VFD
       ├── VFDS-15
           ├── train_resize
               ├── 1.jpg
               ├── 3.jpg
               └── ...
           ├── test_resize
               ├── 2.jpg
               ├── 9.jpg
               └── ...
       └── VFDL-15
           ├── train_resize
               ├── 1.jpg
               ├── 3.jpg
               └── ...
           ├── test_resize
               ├── 2.jpg
               ├── 9.jpg
               └── ...
~~~~     

## Training & Evaluation

To train the our model on Nutrition5K dataset, please run this command:

```train
cd .projects/coarse_to_fine/Nutrition5K/lib
python main.py
```
To train the our model on ECUSTFD dataset, please run this command:

```train
cd .projects/coarse_to_fine/ECUSTFD/lib
python main.py
```
To train the our model on VFD dataset, please run this command:

```train
cd .projects/coarse_to_fine/VFD/lib
python main.py
```

## Evaluation

To evaluate our model on Nutrition5K, run:

```eval
cd .projects/coarse_to_fine/Nutrition5K/lib/evaluation
python compute_eval_statistics.py
```
To evaluate SBF-Net model on ECUSTFD, run:

```eval
cd .projects/coarse_to_fine/ECUSTFD/lib/evaluation
python compute_eval_statistics.py
```
To evaluate SBF-Net model on VFD, run:

```eval
cd .projects/coarse_to_fine//VFD/lib/evaluation
python compute_eval_statistics.py
```


## Pre-trained Models

You can download pretrained models and our datasets here:
[Baidu Netdisk]( https://pan.baidu.com/s/1pf_A0F8rFZzTMi-1Nsp2zg).

## Results

Our model achieves the following performance:

### [Nutrition5K](https://github.com/google-research-datasets/Nutrition5k)

| Nutrition | Calories  | Mass  | Fat  | Carb  | Protein  |
| --------- | :-----: | :-----: | :-----: | :-----: | :-----: |
| pMAE      | 25.0 | 19.7 | 37.4 | 32.3 | 34.5 |

### [ECUSTFD](https://github.com/Liang-yc/ECUSTFD-resized-)

| Food |Bread|Sachima|Fried_wist|Litchi|Mooncake|Plum|Qiwi|Egg|Bun|Mango|
| ---- |:----:| :----: |:----:| :----: | :----:|:----:| :----: |:----:|:----:|:----:|
| pMAE | 16.7| 6.9 |10.1| 7.2 | 18.3 |7.7|11.3|8.3|13.0|5.3|
| **Food**|**Lemon**|**Peach**|**Doughnut**| **Banana** |**Orange**|**Tomato**|**Pear**|**Grape**| **Apple** |**Mean**|
| pMAE | 0.5| 15.9 |8.7| 15.4 | 5.7 |6.2|2.5|12.5|2.7|9.2|

### [VFD](https://drive.google.com/file/d/1CobbDAw_QeZfitBPleZGBnXY0nkntKtw/view?usp=sharin)
VFDS-15

| class | 1 | 2 | 3 | 4 |5|6|7|8|9|10|11|12|13|14|15 | Mean |
| -- | :----: |:----:| :----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| pMAE |12.8|7.4|10.6|9.0|9.4|8.6|8.3|8.1|7.3|7.0|6.5|5.8|6.0|4.9|3.7|6.4|


VFDL-15
| class | 1 | 2 | 3 | 4 |5|6|7|8|9|10|11|12|13|14|15 | Mean |
| -- | :----: |:----:| :----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| pMAE |12.3|10.9|10.9|10.0|7.8|9.8|9.0|6.2|6.0|6.0|4.6|5.8|6.0|5.3|7.2|6.9|