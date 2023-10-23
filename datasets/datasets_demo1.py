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
from torchvision import transforms
from PIL import Image, ExifTags


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

        self.train_list = []
        json_path = self.root + 'train/라벨링데이터/'
        lists = os.listdir(json_path.encode('utf8'))
        from tqdm import tqdm
        for index, l in enumerate(lists):
            if '제품' == l.decode('utf8') or '제품 착용' == l.decode('utf8'):
                folder = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
                for f in tqdm(folder, total=len(folder), desc="제품 및 제품 착용"):
                    json = os.listdir((json_path + l.decode('utf8') + '/' + f.decode('utf8')).encode('utf8'))
                    category = (json_path + l.decode('utf8') + '/' + f.decode('utf8')).split('/')[-1]
                    if category:
                        for idx, j in enumerate(json):
                            if os.path.isfile((json_path + l.decode('utf8') + '/' + f.decode('utf8') + '/' + j.decode(
                                    'utf8')).encode('utf8')):
                                if str(j)[:-1].endswith(".json"):
                                    # if (len(json) - idx) % 100 == 0:
                                    self.train_list.append(
                                            json_path + l.decode('utf8') + '/' + f.decode('utf8') + '/' + j.decode(
                                                'utf8'))

            elif '모델' == l.decode('utf8'):
                json = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
                for idx, j in tqdm(enumerate(json, total=len(json)), desc="모델"):
                    if os.path.isfile((json_path + l.decode('utf8') + '/' + j.decode('utf8')).encode('utf8')):
                        if str(j)[:-1].endswith(".json"):
                            # if (len(json)-idx) % 10 == 0 :
                            self.train_list.append(json_path + l.decode('utf8') + '/' + j.decode('utf8'))

        self.number_samples = len(self.train_list)

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
        # imname = path.split('/')[-1]
        # img.save('./log/sp_results/' + imname + '.jpg')
        return img

    def parsing_mask(self, path):
        try:
            with open(path, 'r', encoding="utf-8") as f:
                annot = json.load(f)
                self.ls = path.decode('utf-8')
                for k, v in annot.items():
                    if k == 'info':
                        img_path = annot[k][0]['image']['path']
                        self.ls = self.ls.split('/')
                        img_path = '/' + self.ls[1] + '/' + self.ls[2] + '/' + self.ls[3] + '/' + self.ls[4] + '/' + \
                                   self.ls[5] + '/' + self.ls[6] + img_path

                        # img_path = img_path.replace('json', 'jpg')
                        # img_path = img_path.replace('JSON', 'jpg')
                        img_path = img_path.replace('JPG', 'jpg')
                        try:
                            img = self.img_rotate_resize(img_path)
                            im = np.asarray(img, np.uint8)

                        except Exception as e:
                            print("not readable image path:", img_path)
                            return None, None, None, None

                    elif k == 'annotation':
                        seg = annot[k][0]['segmentation']

                h, w, _ = im.shape
                org_h = int(annot['info'][0]['image']['height'])
                org_w = int(annot['info'][0]['image']['width'])
                tem_parsing_anno = np.zeros((h, w), dtype=np.int32)

                # for idx in seg:
                #     last_label = []
                #     if int(idx) < 24:
                #         for i in range(1, len(seg[idx]), 2):
                #             last_label.append([int(seg[idx][i - 1]*(3024/org_w)), int(seg[idx][i]*(4032/org_h))])
                #         if len(last_label) <= 0:
                #             print('path : ', self.ls)
                #         tem_parsing_anno = cv2.fillPoly(tem_parsing_anno, [np.array(last_label, np.int32)], [int(idx)])

                # Accessory 데이터의 경우 이중 리스트 형태이므로 코드 수정
                for idx in seg:
                    for n in range(len(seg[idx])):
                        last_label = []
                        if int(idx) < 24:
                            for i in range(1, len(seg[idx][n]), 2):
                                last_label.append(
                                    [int(seg[idx][n][i - 1] * (3024 / org_w)), int(seg[idx][n][i] * (4032 / org_h))])
                                if len(last_label) <= 0:
                                    print('path : ', self.ls)
                            tem_parsing_anno = cv2.fillPoly(tem_parsing_anno, [np.array(last_label, np.int32)],
                                                            [int(idx)])

            return im, tem_parsing_anno, h, w

        except Exception as e:
            print("not readable json path:", path.decode('utf-8'))
            return None, None, None, None

    def __getitem__(self, index):

        train_item = self.train_list[index]
        im, parsing_anno, h, w = self.parsing_mask(train_item.encode('utf8'))

        if type(im) == type(None):
            return None,None,None

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
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


        self.val_list = []
        json_path = self.root + 'test/라벨링데이터/'
        print(json_path)
        lists = os.listdir(json_path.encode('utf8'))
        from tqdm import tqdm
        for index, l in enumerate(lists):
            if '제품' == l.decode('utf8') or '제품 착용' == l.decode('utf8'):
                folder = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
                for f in tqdm(folder, total=len(folder), desc="제품 및 제품 착용"):
                    json = os.listdir((json_path + l.decode('utf8') + '/' + f.decode('utf8')).encode('utf8'))
                    category = (json_path + l.decode('utf8') + '/' + f.decode('utf8')).split('/')[-1]
                    if category:
                        for idx, j in enumerate(json):
                            if os.path.isfile((json_path + l.decode('utf8') + '/'+ f.decode('utf8') + '/' + j.decode('utf8')).encode('utf8')):
                                    if str(j)[:-1].endswith(".json"):
                                        # if (len(json)-idx) % 1000 == 0:
                                        self.val_list.append(json_path + l.decode('utf8') + '/' + f.decode('utf8') + '/' + j.decode('utf8'))

            elif '모델' == l.decode('utf8'):
                json = os.listdir((json_path + l.decode('utf8')).encode('utf8'))
                for idx, j in tqdm(enumerate(json, total=len(json)), desc="모델"):
                    if os.path.isfile((json_path + l.decode('utf8') + '/'+ j.decode('utf8')).encode('utf8')):
                        if str(j)[:-1].endswith(".json"):
                            # if (len(json)-idx) % 20 == 0 :
                            self.val_list.append(json_path + l.decode('utf8') + '/' + j.decode('utf8'))

        self.number_samples = len(self.val_list)


    def __len__(self):
        return len(self.val_list)

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
        # imname = path.split('/')[-1]
        # img.save('./log/sp_results/' + imname + '.jpg')
        return img

    def img_resize(self, path):
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
                        img_path = annot[k][0]['image']['path']
                        self.ls = self.ls.split('/')
                        img_path = '/' + self.ls[1] + '/' + self.ls[2] + '/' + self.ls[3] + '/' + self.ls[4] + '/' + \
                                   self.ls[5] + '/' + self.ls[6] + img_path

                        # img_path = img_path.replace('json', 'jpg')
                        # img_path = img_path.replace('JSON', 'jpg')
                        img_path = img_path.replace('JPG', 'jpg')
                        try:
                            img = self.img_rotate_resize(img_path)
                            im = np.asarray(img, np.uint8)

                        except Exception as e:
                            print("not readable image path:", img_path)
                            return None, None, None, None

                    elif k == 'annotation':
                        seg = annot[k][0]['segmentation']

                h, w, _ = im.shape
                org_h = int(annot['info'][0]['image']['height'])
                org_w = int(annot['info'][0]['image']['width'])
                tem_parsing_anno = np.zeros((h, w), dtype=np.int32)

                # for idx in seg:
                #     last_label = []
                #     if int(idx) < 24:
                #         for i in range(1, len(seg[idx]), 2):
                #             last_label.append([int(seg[idx][i - 1]*(3024/org_w)), int(seg[idx][i]*(4032/org_h))])
                #         if len(last_label) <= 0:
                #             print('path : ', self.ls)
                #         tem_parsing_anno = cv2.fillPoly(tem_parsing_anno, [np.array(last_label, np.int32)], [int(idx)])

                for idx in seg:
                    for n in range(len(seg[idx])):
                        last_label = []
                        if int(idx) < 24:
                            for i in range(1, len(seg[idx][n]), 2):
                                last_label.append([int(seg[idx][n][i - 1]*(3024/org_w)), int(seg[idx][n][i]*(4032/org_h))])
                                if len(last_label) <= 0:
                                    print('path : ', self.ls)
                            tem_parsing_anno = cv2.fillPoly(tem_parsing_anno, [np.array(last_label, np.int32)], [int(idx)])

            return im, tem_parsing_anno, h, w

        except Exception as e:
            print("not readable json path:", path.decode('utf-8'))
            return None, None, None, None

    def __getitem__(self, index):
        val_item = self.val_list[index]
        # Load training image
        im, parsing_anno, h, w = self.parsing_mask(val_item.encode('utf8'))
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
        return batch_input_im, parsing_anno, meta

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