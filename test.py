import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import scores

def test(args):

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = misc.imread(args.img_path)

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset, config_file=args.config_file)
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img = img.astype(float) / 255.0
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model = torch.load(args.model_path)
    model.eval()

    #if torch.cuda.is_available():
    #    model.cuda(0)
    #    images = Variable(img.cuda(0))
    #else:
    images = Variable(img)


    import time
    start_time = time.clock()
    outputs = model(images)
    print('Time: {time}'.format(time=(time.clock() - start_time)))

    pred = outputs[0].cpu().data.numpy()
    if args.label:
        pred = pred[args.label]
    else:
        pred = pred.argmax(0)
    misc.imsave(args.out_path, pred)
    print("Segmentation Mask Saved at: {}".format(args.out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--config_file', nargs='?', type=str, default='./dataroot/segmentation_path_config.json')
    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    parser.add_argument('--label', nargs='?', type=int, help='Label number to write in image mask')
    args = parser.parse_args()
    test(args)
