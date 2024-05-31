

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import utils
import matplotlib.pyplot as plt

from dataset.data import get_unpair_data, get_test_data
from skimage.metrics import structural_similarity as ssim

from models.LCDBNet import LCDBNet


parser = argparse.ArgumentParser(description='Image Deraining using SwinIR')
parser.add_argument('--input_dir', default='G:/low-light_image_enhancement/Our/dataset/LOL', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results/LCDBNet/', type=str, help='Directory for results')
parser.add_argument('--weights', default='checkpoints/lol.pth', type=str,
                    help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


model_restoration = LCDBNet().cuda()


utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration.eval()

datasets = ['eval15']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset)
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    with torch.no_grad():
        psnr_yuv = []
        ssim_yuv = []
        psnr_fn = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            target = data_test[0].cuda()
            input_ = data_test[1].cuda()
            filenames = data_test[2]

            with torch.no_grad():
                Lum, Chrom, restored = model_restoration(input_)

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(0)
            restored = cv2.cvtColor(restored, cv2.COLOR_YCrCb2BGR)
            target = target.permute(0, 2, 3, 1).cpu().detach().numpy().squeeze(0)
            target = cv2.cvtColor(target, cv2.COLOR_YCrCb2BGR)

            cur_psnr = utils.PSNR(target, restored)
            print("[Image: %s PSNR: %.4f]" % (filenames[0], cur_psnr))

            psnr_yuv.append(cur_psnr)

            cv2.imwrite((os.path.join(result_dir, filenames[0] + '.png')), restored)

        psnr_val_yuv = np.mean(psnr_yuv)
        print("[Testset: %s PSNR: %.4f ]" % (dataset, psnr_val_yuv))
