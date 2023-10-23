import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from datetime import datetime

now = datetime.datetime.now()
print(now)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_HOME"] = '/usr/local/cuda-10.1'

from datasets.sweetk_datasets import SWEETKDataValSet
import networks
from utils.miou import compute_mean_ioU_sweetk_once, AP, ElevenPointInterpolatedAP
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing
from collections import OrderedDict

SWEETK_ACCESORY = ["Background","Hat","Glasses frame","Sunglasses","Necklace","Earrings"]
SWEETK_SHAPELESS = ["Background","Face","Left-arm","Right-arm","Left-leg", "Right-leg",
                    "Normal_top","Normal_bottom","Coat","jacket","Jumper","Padding",
                    "vest","Cardigan","Blouse","Top","T-shirt","shirts","Sweater",
                    "Pants","Skirt","Dress","jumpsuit"]

def get_arguments():
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")
    parser.add_argument("--arch", type=str, default='resnet101')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
    parser.add_argument("--save-results", action="store_true", help="whether to save the results.")
    parser.add_argument("--flip", action="store_true", help="random flip during the test.")
    parser.add_argument("--multi-scales", type=str, default='1', help="multiple scales during the test")
    parser.add_argument("--ignore-label", type=int, default=255)

    # # ###################################################SHAPELESS############################################################
    # parser.add_argument("--data-dir", type=str, default='/workspace/data/dataset/20221230/K-Deep-Fashion-Shapeless/쉐이프리스 의류 및 포즈 데이터/')
    # parser.add_argument("--num-classes", type=int, default=23)
    # parser.add_argument("--labels_num", type=int, default=0)
    # parser.add_argument("--model-restore", type=str, default='/workspace/log_shapeless/checkpoint_90.pth.tar')
    # # ##########################################################################################################################

    #################################################ACCESSORIES############################################################
    parser.add_argument("--data-dir", type=str, default='/workspace/data/old2/K-Deep-Fashion/패션 액세서리 착용 데이터/')
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--labels_num", type=int, default=1)
    parser.add_argument("--model-restore", type=str, default='/workspace/log_accessory/checkpoint_100.pth.tar')
    ########################################################################################################################

    return parser.parse_args()

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def multi_scale_testing(model, batch_input_im, crop_size=[473, 473],  multi_scales=[1]):
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)
    interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1]
        output = parsing_output
        output = interp(output)
        ms_outputs.append(output)
    ms_fused_parsing_output = torch.stack(ms_outputs)[0]
    ms_fused_parsing_output = ms_fused_parsing_output
    ms_fused_parsing_output = ms_fused_parsing_output.permute(0, 2, 3, 1)
    parsing = torch.argmax( ms_fused_parsing_output, dim=3)
    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()

    return parsing, ms_fused_parsing_output

def make_category_list(category):
    for key, val in category.items():
        key = key.replace(' ', '')
        globals()[key] = []

def miou_append(miou) :
    for key, val in miou.items():
        key = key.replace(' ', '')
        globals()[key].append(int(val))

def miou_mean(miou_once) :
    categories=[]
    for key, _ in miou_once.items():
        key = key.replace(' ', '')



        mean_cate = []
        for i in globals()[key]:
            if i != 0:
                mean_cate.append(i)
        if len(mean_cate) > 0:
            mean_cate = sum(mean_cate) / len(mean_cate)
        else:
            mean_cate = 0
        categories.append((key, mean_cate))
    t_miou = []
    for key, val in categories :
        acc_list = ['MeanIU', 'Meanaccuracy', 'Pixelaccuracy']
        if key not in acc_list:
            t_miou.append(val)
    mean_iou = sum(t_miou) / len(t_miou)
    categories.append(('MeanIU', mean_iou))
    categories = OrderedDict(categories)
    return categories

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    multi_scales = [float(i) for i in args.multi_scales.split(',')]
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    cudnn.enabled = True
    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]
    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=None)
    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])
    if INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # Data loader
    lip_test_dataset = SWEETKDataValSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    num_samples = len(lip_test_dataset)
    print('Totoal testing sample numbers: {}'.format(num_samples))
    testloader = data.DataLoader(lip_test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Load model weight
    state_dict = torch.load(args.model_restore)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    sp_results_dir = os.path.join(args.log_dir, 'sp_results')
    if not os.path.exists(sp_results_dir):
        os.makedirs(sp_results_dir)
    palette = get_palette(20)
    img_names = []
    Iou = []
    mean_accuracy = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image, label, meta = batch
            if (len(image.shape) > 4):
                image = image.squeeze()
            parsing, logits = multi_scale_testing(model, image.cuda(), crop_size=input_size, multi_scales=multi_scales)
            for index, p in enumerate(parsing) :
                im_name = meta['name'][index]
                c = meta['center'].numpy()[index]
                s = meta['scale'].numpy()[index]
                w = meta['width'].numpy()[index]
                h = meta['height'].numpy()[index]
                parsing_preds = [(p, label[index].unsqueeze(0))]
                if args.labels_num == 0:
                    LABELS = SWEETK_SHAPELESS
                else:
                    LABELS = SWEETK_ACCESORY
                miou_once = compute_mean_ioU_sweetk_once(im_name, parsing_preds, s, c, args.num_classes,
                                   args.data_dir, LABELS, input_size)
                if (idx == 0) & (index == 0):
                    make_category_list(miou_once)
                miou_append(miou_once)
                img_name = im_name.split('/')[-1]
                img_name = img_name.rstrip('.json')
                img_names.append(img_name)
                Iou.append(round(miou_once['Mean IU'], 5))
                mean_accuracy.append(round(miou_once['Mean accuracy'], 5))

                if args.save_results:
                    parsing_result = transform_parsing(p, c, s, w, h, input_size)
                    imname = im_name.split('/')[-1]
                    # imname = ''.join(im_name.split('/')[-2:])
                    parsing_result_path = os.path.join('./log/sp_results', imname + '.png')
                    output_im = PILImage.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                    output_im.putpalette(palette)
                    output_im.save(parsing_result_path)

    mIoU = miou_mean(miou_once)
    log = pd.DataFrame({'img_name': img_names, 'IoU': Iou, 'Mean accuracy': mean_accuracy})
    log.to_csv('/workspace/evaluate_log.csv', index=False, encoding='utf-8')
    print(mIoU)
    print("AP50 : " + str(AP(Iou[1:], 50.0)))
    print("AP75 : " + str(AP(Iou[1:], 75.0)))
    print("AP90 : " + str(AP(Iou[1:], 90.0)))
    return

if __name__ == '__main__':
    main()
