from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
from models.Swin_main import *
import argparse
import numpy as np
import cv2
#from skimage import img_as_ubyte


def main(args):
    # Init model

    device = torch.device("cuda")
    model = DCFM()
    model = model.to(device)
    try:
        modelname = os.path.join(args.param_root, 'best_ep187_Smeasure0.7631.pth')
        dcfmnet_dict = torch.load(modelname)
        print('loaded', modelname)
    except:
        dcfmnet_dict = torch.load(os.path.join(args.param_root, 'best_ep187_Smeasure0.7631.pth'))

    model.to(device)
    model.dcfmnet.load_state_dict(dcfmnet_dict)
    model.eval()
    model.set_mode('test')

    tensor2pil = transforms.ToPILImage()
    for testset in ['RGBD_CoSeg183', 'RGBD_CoSal1k', 'RGBD_CoSal150']:
        if testset == 'CoCA':
            test_img_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoCA/images/'
            test_dep_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoCA/depths/'
            test_gt_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoCA/gts/'
            saved_root = os.path.join(args.save_root, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoSOD3k/images/'
            test_dep_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoSOD3k/depths/'
            test_gt_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoSOD3k/gts/'
            saved_root = os.path.join(args.save_root, 'CoSOD3k')
        elif testset == 'CoSal2015':
            test_img_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoSal2015/images/'
            test_dep_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoSal2015/depths/'
            test_gt_path = '/hy-tmp/DCFM-CoSOD_Depth/Datasets_depth/Depth_test_dataets/CoSal2015/gts/'
            saved_root = os.path.join(args.save_root, 'CoSal2015')
        elif testset == 'RGBD_CoSeg183':
            test_img_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSeg183/images/'
            test_dep_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSeg183/depths/'
            test_gt_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSeg183/gts/'
            saved_root = os.path.join(args.save_root, 'RGBD_CoSeg183')
        elif testset == 'RGBD_CoSal1k':
            test_img_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSal1k/images/'
            test_dep_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSal1k/depths/'
            test_gt_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSal1k/gts/'
            saved_root = os.path.join(args.save_root, 'RGBD_CoSal1k')
        elif testset == 'RGBD_CoSal150':
            test_img_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSal150/images/'
            test_dep_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSal150/depths/'
            test_gt_path = './Datasets_depth/Depth_test_dataets/RGBD_CoSal150/gts/'
            saved_root = os.path.join(args.save_root, 'RGBD_CoSal150')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)

        test_loader = get_loader(
            test_img_path, test_dep_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=0,
            pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            dep = batch[1].to(device).squeeze(0)
            gts = batch[2].to(device).squeeze(0)
            subpaths = batch[3]
            ori_sizes = batch[4]
            scaled_preds= model(inputs,dep, gts)
            scaled_preds = torch.sigmoid(scaled_preds[-1])
            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)
            num = gts.shape[0]
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--param_root', default='./temp', type=str, help='model folder')
    parser.add_argument('--save_root', default='./CoSODmaps/pred', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)



