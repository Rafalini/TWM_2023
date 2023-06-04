from __future__ import division

from darknet import Darknet, parse_cfg, create_modules
import torch
from torch.autograd import Variable
import numpy as np
import cv2


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_

#model building
blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))
#random prediction
model = Darknet("cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print(pred)
#weights read
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
pred = model(inp, torch.cuda.is_available())
print(pred)