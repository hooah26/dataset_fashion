#!export LANG=C.UTF-8
#!/opt/conda/bin/python3.6
# -*- encoding: UTF-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   datasets.py
@Time    :   8/4/19 3:35 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""
from random import randint
from PIL import Image as PILImage
import os
import numpy as np
import random
import torch
import cv2
import json
import glob
from torch.utils import data
from utils.transforms import get_affine_transform
import pandas as pd
from torchvision import transforms
from PIL import Image, ExifTags



class FasterSWEETKDataset(data.Dataset):
    def __init__(self):
        file_path = "/workspace/train_dataset.csv"
        self.df = pd.read_csv(file_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        sample = self.df[idx]
        input, label_parsing, meta = sample.values

        return input, label_parsing, meta


class SWEETKDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.transform = transform
        self.dataset = dataset

        # list_path = os.path.join(self.root, self.dataset + '_id.txt')
        # train_list = [i_id.strip() for i_id in open(list_path)]
        #train_list = []
        self.train_list = {}
        # path = self.root + 'json/'
        # list = os.listdir(path)
        # encoding=utf8
        json_path = self.root + '라벨링데이터/train/'
        lists = os.listdir(json_path.encode('utf8'))
        from tqdm import tqdm
        idx = 0
        for index, l in enumerate(lists):
            if '제품' == l.decode('utf8') or '제품 착용' == l.decode('utf8'):
                folder = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
                for f in tqdm(folder, total=len(folder), desc="제품종류"):
                    json = os.listdir((json_path + l.decode('utf8') + '/' + f.decode('utf8')).encode('utf8'))
                    for j in json:
                        if os.path.isfile((json_path + l.decode('utf8') + '/'
                                           + f.decode('utf8') + '/'
                                           + j.decode('utf8')).encode('utf8')):
                            # train_list.append(
                            # json_path + l.decode('utf8') + '/' + f.decode('utf8') + '/' + j.decode('utf8'))
                            if index % 3 == 0:
                                self.train_list[idx] = json_path + l.decode('utf8') + '/' + f.decode(
                                    'utf8') + '/' + j.decode('utf8')
                                idx += 1
                            index += 1
            elif '모델' == l.decode('utf8'):
                json = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
                for j in tqdm(json, total=len(json), desc="모델"):
                    if os.path.isfile((json_path + l.decode('utf8') + '/'
                                       + j.decode('utf8')).encode('utf8')):
                        # train_list.append(
                        # json_path + l.decode('utf8') + '/' + j.decode('utf8'))
                        if index % 3 == 0:
                            self.train_list[idx] = json_path + l.decode('utf8') + '/' + j.decode('utf8')
                            idx += 1
                        index += 1

        #self.train_list = train_list
        self.number_samples = len(self.train_list.keys())
        #self.train_list = {idx:item for idx, item in enumerate(tqdm(train_list, total = len(train_list), desc="convert from list to dict"))}

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def imgfile_check(self, path):
        # jpg check
        if os.path.exists(path):
            return path
        elif os.path.exists(path.replace('jpg','JPG')):
            return path.replace('jpg','JPG')
        else:
            print(path)


    def parsing_mask(self, path):
        try:
            with open(path, 'r', encoding="utf-8") as f:
                annot = json.load(f)
                self.ls = path.decode('utf-8')
                for k, v in annot.items():
                    if k == 'info':
                        img_path = annot[k][0]['image']['path']
                        self.ls = self.ls.split('/')
                        img_path = '/'+self.ls[1]+'/'+self.ls[2]+'/'+self.ls[3]+'/'+ self.ls[4]+'/'+self.ls[5]+'/'+self.ls[6]+img_path
                        # img_path = path.decode('utf8').replace('라벨링데이터','원천데이터')
                        # img_path = img_path.replace('train/', '')
                        # img_path = img_path.replace('json', 'jpg')
                        # img_path = img_path.replace('JSON', 'jpg')
                        # img_path = self.imgfile_check(img_path)

                    elif k == 'annotation':
                        seg = annot[k][0]['segmentation']
                        # for i in seg:
                        #     if type(seg[i]) is str :
                        #         result = []
                        #         data = seg[i].split(",")
                        #         for idx, d in enumerate(data):
                        #             if idx == 0:
                        #                 result.append(float(d[1:]))
                        #             elif idx == len(data) - 1:
                        #                 result.append(float(d[:-1]))
                        #             else:
                        #                 result.append(float(d))
                        #         seg[i] = result
                # img_path = img_path.replace('JPG','jpg')
                # print('1 : ' + self.root[:-1] + img_path)
                # print('1 : ' + path.decode('utf8'))

                # img_array = np.fromfile( (self.root[:-1] + img_path).encode('utf8'), np.uint8)

                    try:
                        img_array = np.fromfile(img_path, np.uint8)#.encode('utf8'), np.uint8)
                        im = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    except Exception as e:
                        print("not readable image path:",img_path)
                        return None, None, None, None
                # im = cv2.imread( self.root[:-1] + img_path, cv2.IMREAD_COLOR)
                #     print("Result  :", self.root[:-1] + img_path)

                h, w, _ = im.shape
                parsing_anno = np.zeros((h, w), dtype=np.int32)

                for idx in seg:
                    last_label = []
                    if int(idx) < 16:
                        for i in range(1, len(seg[idx]), 2):
                            last_label.append([int(seg[idx][i - 1]), int(seg[idx][i])])
                        if len(last_label) <= 0:
                            print('path : ', self.ls)
                        parsing_anno = cv2.fillPoly(parsing_anno, [np.array(last_label, np.int32)], [int(idx)])
            return im, parsing_anno, h, w

        except Exception as e:
            print("not readable json path:",self.ls)
            return None, None, None, None

    def __getitem__(self, index):

        train_item = self.train_list[index]

        # im_path = os.path.join(self.root, self.dataset + '_images', train_item + '.jpg')
        # parsing_anno_path = os.path.join(self.root, self.dataset + '_segmentations', train_item + '.png')
        #
        # print(train_item)
        im, parsing_anno, h, w = self.parsing_mask(train_item.encode('utf8'))

        if type(im) == type(None):
            return None,None,None

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
            # parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
            if self.dataset == 'train' or self.dataset == 'trainval':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]
                    person_center[0] = im.shape[1] - person_center[0] - 1
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)


        meta = {
            'name': train_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset == 'val' or self.dataset == 'test':
            return input, meta
        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, meta



class SWEETKDataValSet(data.Dataset):
    def __init__(self, root, dataset='val', crop_size=[473, 473], transform=None, flip=False):
        self.root = root
        self.crop_size = crop_size
        self.transform = transform
        self.flip = flip
        self.dataset = dataset
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)


        # list_path = os.path.join(self.root, self.dataset + '_id.txt')
        # val_list = [i_id.strip() for i_id in open(list_path)]
### 딕셔너리 val_list 추가 방법 ###
        # self.val_list = {}
        # json_path = self.root + '라벨링데이터/'
        # print(json_path)
        # lists = os.listdir(json_path.encode('utf8'))
        # from tqdm import tqdm
        # idx = 0
        # for index, l in enumerate(lists):
        #     if '제품' == l.decode('utf8') or '제품 착용' == l.decode('utf8'):
        #         folder = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
        #         for f in tqdm(folder, total=len(folder), desc="제품종류"):
        #             json = os.listdir((json_path + l.decode('utf8') + '/' + f.decode('utf8')).encode('utf8'))
        #             for j in json:
        #                 if os.path.isfile((json_path + l.decode('utf8') + '/'+ f.decode('utf8') + '/' + j.decode('utf8')).encode('utf8')):
        #                     self.val_list[idx] = json_path + l.decode('utf8') + '/' + f.decode('utf8') + '/' + j.decode('utf8')
        #                     index += 1
        #     elif '모델' == l.decode('utf8'):
        #         json = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
        #         for j in tqdm(json, total=len(json), desc="모델"):
        #             if os.path.isfile((json_path + l.decode('utf8') + '/'
        #                                + j.decode('utf8')).encode('utf8')):
        #                 self.val_list[idx] = json_path + l.decode('utf8') + '/' + j.decode('utf8')
        #                 index += 1
        # self.number_samples = len(self.val_list.keys())
### 딕셔너리 val_list 추가 방법 ### 끝
        self.val_list = []
        json_path = self.root + '라벨링데이터/'
        print(json_path)
        lists = os.listdir(json_path.encode('utf8'))
        from tqdm import tqdm
        for index, l in enumerate(lists):
            # if '제품' == l.decode('utf8') or '제품 착용' == l.decode('utf8'):
            if '제품 착용' == l.decode('utf8'):
                folder = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
                for f in tqdm(folder, total=len(folder), desc="제품종류"):
                    json = os.listdir((json_path + l.decode('utf8') + '/' + f.decode('utf8')).encode('utf8'))
                    category = (json_path + l.decode('utf8') + '/' + f.decode('utf8')).split('/')[-1]####### 카테고리 찾기
                    if category == '가디건':
                        for idx, j in enumerate(json):
                            if os.path.isfile((json_path + l.decode('utf8') + '/'+ f.decode('utf8') + '/' + j.decode('utf8')).encode('utf8')):
                                    if str(j)[:-1].endswith(".json"):
                                        if (len(json)-idx) % 500 == 0:
                                          self.val_list.append(json_path + l.decode('utf8') + '/' + f.decode('utf8') + '/' + j.decode('utf8'))

            # elif '모델' == l.decode('utf8'):
            #     json = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
            #     for idx, j in tqdm(enumerate(json, total=len(json)), desc="모델"):
            #         if os.path.isfile((json_path + l.decode('utf8') + '/'+ j.decode('utf8')).encode('utf8')):
            #             if str(j)[:-1].endswith(".json"):
            #                 # if (len(json)-idx) % 20 == 0 :
            #                 self.val_list.append(json_path + l.decode('utf8') + '/' + j.decode('utf8'))

        self.number_samples = len(self.val_list)


    def __len__(self):
        return len(self.val_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def get_list(self):
        return self.val_list


    def parsing_json(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            annot = json.load(f)
            img_path = ''
            for k, v in annot.items():
                if k == 'info':
                    img_path = annot[k][0]['image']['path']
                elif k == 'annotation':
                    seg = annot[k][0]['segmentation']
            print(img_path)
            img_path = img_path.replace('JPG', 'jpg')
            img_array = np.fromfile((self.root[:-1] + img_path).encode('utf8'), np.uint8)
            im = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            h, w, _ = im.shape
        return im, h, w

    def imgfile_check(self, path):
        # jpg check
        if os.path.exists(path):
            return path
        elif os.path.exists(path.replace('jpg','JPG')):
            return path.replace('jpg','JPG')
        else:
            print(path)

    def img_rotate_resize(self, path):
        image = Image.open(path)
        img_exif = image.getexif()
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        del img_exif[orientation]
        img = image.resize((3024, 4032), Image.LANCZOS)
        imname = path.split('/')[-1]
        # imname = ''.join(path.split('/')[-2:])
        img.save('./log/sp_results/' + imname + '.jpg')
        return img

##############cv2이용##################
    def img_resize(self, path):
        ###############################################
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, dsize=(3024, 4032), interpolation=cv2.INTER_LINEAR)
        # img = cv2.resize(img, dsize=(302, 403), interpolation=cv2.INTER_AREA)

        # imname = path.split('/')[-1]
        # cv2.imwrite('./log/sp_results/' + imname + '.jpg', img)
        return img

    def parsing_mask(self, path):
        try:
            with open(path, 'r', encoding="utf-8") as f:
                annot = json.load(f)
                self.ls = path.decode('utf-8')
                for k, v in annot.items():
                    if k == 'info':
                        ########################org서버용 #####################################
                        img_path = annot[k][0]['image']['path']
                        self.ls = self.ls.split('/')
                        ########################shapeless #####################################
                        img_path = '/' + self.ls[1] + '/' + self.ls[2] + '/' + self.ls[3] + '/' + self.ls[4] + '/' + \
                                   self.ls[5] + '/' + self.ls[6] + img_path
                        #######################악세서리 #####################################
                        # img_path = '/' + self.ls[1] + '/' + self.ls[2] + '/' + self.ls[3] + '/' + self.ls[4] + '/' + \
                        #            self.ls[5] + img_path

                        img_path = img_path.replace('json', 'jpg')
                        img_path = img_path.replace('JSON', 'jpg')
                        # img_path = self.imgfile_check(img_path)

                        #########################연습용 도커####################################
                        # img_path = self.ls.replace('라벨링데이터','원천데이터')
                        # img_path = img_path.replace('json', 'jpg')
                        # img_path = img_path.replace('JSON', 'jpg')
                        # img_path = self.imgfile_check(img_path)
                        try:
                        ####################### org_code#############################
                            # img_array = np.fromfile(img_path, np.uint8)
                            # im = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

                        #########################re_size#################################
                            # img = self.img_resize(img_path)
                            # im = np.asarray(img, np.uint8)

                            img = self.img_rotate_resize(img_path)
                            im = np.asarray(img, np.uint8)

                        except Exception as e:
                            print("not readable image path:", img_path)
                            return None, None, None, None

                    elif k == 'annotation':
                        seg = annot[k][0]['segmentation']

                h, w, _ = im.shape
                ###############################사이즈조절시##################################################
                org_h = int(annot['info'][0]['image']['height'])
                org_w = int(annot['info'][0]['image']['width'])
                # tem_parsing_anno = np.zeros((h, w), dtype=np.int32)
                ############################################################################################
                tem_parsing_anno = np.zeros((h, w), dtype=np.int32)

                color_list = []
                for i in range(23):
                    b = int(randint(1, 255))
                    g = int(randint(1, 255))
                    r = int(randint(1, 255))
                    bgr = (int(b), int(g), int(r))
                    color_list.append(bgr)

                for idx in seg:
                    for n in range(len(seg[idx])):
                        last_label = []
                        if int(idx) < 24:
                            for i in range(1, len(seg[idx][n]), 2):
                                ################################org_code#########################################
                                # last_label.append([int(seg[idx][i - 1]), int(seg[idx][i])])
                                #################################사이즈 조절시######################################
                                last_label.append([int(seg[idx][n][i - 1]*(3024/org_w)), int(seg[idx][n][i]*(4032/org_h))])
                                # last_label.append([int(seg[idx][i - 1] * (302 / org_w)), int(seg[idx][i] * (403 / org_h))])
                            if len(last_label) <= 0:
                                print('path : ', self.ls)
                            # tem_parsing_anno = cv2.fillPoly(tem_parsing_anno, [np.array(last_label, np.int32)], [int(idx)])
                            img2 = cv2.fillPoly(tem_parsing_anno, [np.array(last_label, np.int32)], color=color_list[int(idx)])


                    image11 = PILImage.fromarray(np.asarray(img2, dtype=np.uint8))
                    imname1 = img_path.split('/')[-1] + '1.png'
                    img_name1 = os.path.join('./log/sp_results', imname1)
                    image11.save(img_name1)

                parsing_anno = tem_parsing_anno
            return im, parsing_anno, h, w

        except Exception as e:
            print("not readable json path:", path.decode('utf-8'))
            return None, None, None, None

    def __getitem__(self, index):
        val_item = self.val_list[index]
        # Load training image
        im, anno, h, w = self.parsing_mask(val_item.encode('utf8'))
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        if self.transform:
            input = self.transform(input)
        flip_input = input.flip(dims=[-1])
        if self.flip:
            batch_input_im = torch.stack([input, flip_input])
        else:
            batch_input_im = input

        meta = {
            'name': val_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }
        return batch_input_im, anno, meta

def collate_fn(batch):
    try:
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch)>0:
            return torch.utils.data.dataloader.default_collate(batch)
        else:
            return None
    except Exception as e:
        print("collate fn Error occured : ", e)
        return None