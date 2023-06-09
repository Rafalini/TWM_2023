from __future__ import division
from utils.reshapeOutput import *
from models.darknet import Darknet

batch_size = int(1)
confidence = float(0.5)
nms_thesh = float(0.4)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("cfg/coco.names")

#Set up the neural network
print("Loading network.....")
model = Darknet('cfg/yolov4.cfg')
model.load_weights('cfg/yolov4.weights')
print("Network successfully loaded")

model.net_info["height"] = 832
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()
#Set the model in evaluation mode
model.eval()

input_img = cv2.imread('data/dog-cycle-car.png')
input_tensor = prep_image(input_img, inp_dim)

if CUDA:
    input_tensor = input_tensor.cuda()
with torch.no_grad():
    prediction = model(Variable(input_tensor), CUDA)

prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)
print(prediction)
objs = [classes[int(x[-1])] for x in prediction]
print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()
    
