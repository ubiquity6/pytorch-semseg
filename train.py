import time
import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tensorboard_logger import configure, log_value

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.loss import cross_entropy2d
from ptsemseg.metrics import scores

def train(args):

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset, config_file=args.config_file)
    loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    n_classes = loader.n_classes

    # must use 1 worker for AWS sagemaker without ipc="host" or larger shared memory size
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=1, shuffle=True)

    # Setup Model
    model = get_model(args.arch, n_classes)

    # Setup log dir / logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    configure(args.log_dir)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.9, weight_decay=5e-4)

    step = 0
    for epoch in range(args.n_epoch):
        start_time = time.time()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = cross_entropy2d(outputs, labels)

            loss.backward()
            optimizer.step()

            log_value('Loss', loss.data[0], step)
            step += 1

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]),
                      flush=True)

        end_time = time.time()
        print('Epoch run time: %s' % (end_time - start_time))
        torch.save(model, args.log_dir + "{}_{}_{}_{}.pt".format(args.arch, args.dataset, args.feature_scale, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--config_file', nargs='?', type=str, default='./dataroot/segmentation_path_config.json')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--log_dir', nargs='?', type=str,
                        default='./logs/segmentation-{time}/'.format(time=int(time.time())))
    args = parser.parse_args()
    train(args)
