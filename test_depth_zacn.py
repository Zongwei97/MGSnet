import torch
import torch.nn.functional as F
import imageio
import os, argparse
from data_RGB import test_dataset
from model.VGG16_depth import vgg16_depth_b_zacn as vgg16_depth

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

model = vgg16_depth()
model.load_state_dict(torch.load('path of weight'))

model.cuda()
model.eval()

test_datasets = ['DES', 'DUT', 'LFSD', 'NLPR_TEST', 'NJU2K_TEST', 'SIP', 'SSD', 'STERE']

for dataset in test_datasets:

    save_path = './results/VGG16_simple_init_zacn/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = 'Your TestingSet path'+ dataset + '/RGB/'
    depth_root = 'Your TestingSet path'+ dataset + '/depth/'
    gt_root = 'Your TestingSet path'+ dataset + '/GT/'

    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, d, name = test_loader.load_data()
        image = image.cuda()
        d = d.cuda()
        res = model(image, d)
        res = F.softmax(res, dim=1)

        res = res[0][1]
        res = res.data.cpu().numpy().squeeze()

        imageio.imwrite(save_path+name.split('/')[-1], res)
