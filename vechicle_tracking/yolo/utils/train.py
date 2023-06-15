# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import shutil
import time
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.autograd import Variable

from utils.data import train_batches
from utils.loss import loss, convert2viz
from models.model import YOLO
import matplotlib.pyplot as plt
import numpy as np


def train(args):
    torch.cuda.empty_cache()
    print('Dataset of instance(s) and batch size is {}'.format(args['batch_size']))
    # vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')
    # model = YOLO(vgg.features)
    model = YOLO()
    if args['use_cuda']:
        model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    best = 1e+30

    for epoch in range(1, args['epochs']+1):
        l = train_epoch(epoch, model, optimizer, args)

        is_best = l < best
        best = min(l, best)
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'optimizer' : optimizer.state_dict(),
    #     }, is_best)
    # checkpoint = torch.load('./model_best.weights')
    # state_dict = checkpoint['state_dict']

def train_epoch(epoch, model, optimizer, args):
    losses = 0.0
    for i, (x, y) in enumerate(train_batches(args['batch_size'], use_cuda=args['use_cuda']), 1):
        optimizer.zero_grad()
        y_pred = model(x)
        l = loss(y_pred, y, use_cuda=args['use_cuda'])
        l.backward()
        optimizer.step()
        losses += l.item()
    print("Epoch: {}, Ave loss: {}".format(epoch, losses / i))
    time.sleep(20)
    return losses / i

def test_epoch(model, use_cuda=False, jpg=None):
    if jpg is None:
        x = torch.randn(1, 3, 480, 640)
    else:
        img = plt.imread(jpg) / 255.
        x = torch.from_numpy(np.transpose(img, (2, 0, 1)))

    x = Variable(x, requires_grad=False)

    if use_cuda:
        x = x.cuda()

    y = model(x)
    upperleft, bottomright, classes, confs = convert2viz(y)


def pretrain():
    pass

def save_checkpoint(state, is_best, filename='checkpoint.weights'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.weights')



