import os
import shutil
import random
import splitfolders

splitfolders.ratio('/workspace/data/old2/K-Deep-Fashion/패션 액세서리 착용 데이터/라벨링데이터/제품 착용',
                   output='/workspace/data/old2/K-Deep-Fashion/패션 액세서리 착용 데이터/라벨링데이터',
                   seed=77, ratio=(0.8,0.1,0.1))

# def full_num(num):
#     fir = random.randrange(0,5)
#     while True:
#         sec = random.randrange(0, 4)
#         if fir != sec:
#             break
#     num = [0,0,0,0,0]
#     num[fir] = 1
#     num[sec] = 2
#
#     return num

# name = '/제품 착용/귀걸이/'
# path = 'data/sweetk/1.Dataset/라벨링데이터/train' + name
# path_val = 'data/sweetk/1.Dataset/라벨링데이터/val' + name
# path_test = 'data/sweetk/1.Dataset/라벨링데이터/test' + name
#
# list = os.listdir(path)
# list.sort()

# num = []
# check = 0
# for i in list:
#     if len(num) <= 0:
#         num = full_num(num)
#
#     check = num[0]
#     del num[0]
#
#     if check == 0:
#         print(i)
#     elif check == 1:
#         shutil.move(path+i , path_val+i)
#     elif check == 2:
#         shutil.move(path+i , path_test+i)


# list1 = os.listdir(path)
# list2 = os.listdir(path_test)
# list3 = os.listdir(path_val)
# print(len(list1))
# print(len(list2))
# print(len(list3))


