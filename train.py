import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from data import *
import utils
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--epoch', default=120, type=int)
parser.add_argument('--gamma', default=0.3, type=int)
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        pass
    elif args.dataset == 'VOC':
        cfg = voc
        if not args.resume:
            dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        else:
            dataset = VOCDetection(root=args.dataset_root, transform=BaseTransform(300, (104, 117, 123)))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    if args.cuda:
        ssd_net = ssd_net.cuda()
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.AdamW(ssd_net.parameters(), lr=args.lr)
    criterion = MultiBoxLoss(
        cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
        False, args.cuda
    )

    # loss counters
    print('Loading the dataset...')

    print('Training SSD on:', dataset.name)

    step_index = 0

    data_loader = data.DataLoader(
        dataset, args.batch_size,
        num_workers=args.num_workers,
        shuffle=True, collate_fn=detection_collate,
        pin_memory=True
    )

    ssd_net.train()
    for epoch in range(args.epoch):
        loss_average = utils.AverageMeter()
        loss_l_average = utils.AverageMeter()
        loss_c_average = utils.AverageMeter()
        loss_ciou_average = utils.AverageMeter()
        for i, (images, targets) in enumerate(data_loader):
            iteration = epoch * len(data_loader) + i
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # load train data
            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda().requires_grad_(False) for ann in targets]
            else:
                images = images
                targets = [ann.requires_grad_(False) for ann in targets]

            # forward
            out = ssd_net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_ciou = criterion(out, targets)
            loss = loss_l + loss_c + loss_ciou
            loss.backward()
            optimizer.step()

            loss_average.update(loss.item(), args.batch_size)
            loss_l_average.update(loss_l.item(), args.batch_size)
            loss_c_average.update(loss_c.item(), args.batch_size)
            loss_ciou_average.update(loss_ciou.item(), args.batch_size)

            if iteration % 100 == 0:
                print('iter ' + repr(iteration) + \
                ' || Loss: %.4f' % (loss_average.avg) + \
                ' || LossLoc: %.4f' % (loss_l_average.avg) + \
                ' || LossConf: %.4f' % (loss_c_average.avg) + \
                ' || LossCIOU: %.4f' % (loss_ciou_average.avg)
                , end='\n')

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(
                    ssd_net.state_dict(), 
                    './weights/ssd_mobilenetv1_' + repr(iteration) + '.pth'
                )
        torch.cuda.empty_cache()


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    train()
