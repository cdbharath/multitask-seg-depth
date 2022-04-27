import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import sys
sys.path.append('../')
import scipy.io
import glob

from mnet.model import MNET
from torch.utils.data import DataLoader
from dataset import NYUDDataset
from torchvision.transforms import transforms
from utils import Normalise, ToTensor

import torch
from torch.autograd import Variable

# Pre-processing and post-processing constants #
DEPTH_COEFF = 5000. # to convert into metres
HAS_CUDA = torch.cuda.is_available()

IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

MAX_DEPTH = 8.
MIN_DEPTH = 0.

NUM_CLASSES = 40
NUM_TASKS = 2 # segm + depth

def prepare_img(img):
    return (img/255 - IMG_MEAN) / IMG_STD

model = MNET(NUM_TASKS, NUM_CLASSES)

if HAS_CUDA:
    model.cuda()
model.eval()

ckpt = torch.load('checkpoint.pth')
model.load_state_dict(ckpt)
ckpt = torch.load('weights/ExpNYUD_joint.ckpt')
model.enc.load_state_dict(ckpt["state_dict"], strict=False)
model.dec.load_state_dict(ckpt["state_dict"], strict=False)


# Inference by image
# img_path = 'examples/img_5001.png'
# img = np.array(Image.open(img_path))
# gt_segm = np.array(np.array(scipy.io.loadmat('examples/img_5001.mat')["segmentation"]))
# gt_depth = np.array(Image.open('examples/img_5001_depth.png'))

# Inference by dataset
img_paths = sorted(glob.glob("./nyud/data/images/*"))
seg_paths = sorted(glob.glob("./nyud/segmentation/*"))
depth_paths = sorted(glob.glob("./nyud/data/depth/*"))
test_img_paths = img_paths[int(0.8*len(img_paths)):]
test_seg_paths = seg_paths[int(0.8*len(img_paths)):]
test_depth_paths = depth_paths[int(0.8*len(img_paths)):] 

img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])

print("[INFO]: Loading data")
dataset = NYUDDataset(test_img_paths, test_seg_paths, test_depth_paths)
sample = dataset[np.random.randint(0, len(dataset) - 1)]

with torch.no_grad():

    # Inference by image
    # img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
    # if HAS_CUDA:
    #     img_var = img_var.cuda()

    # depth, segm = model(img_var)

    # Inference by data
    gt_segm = sample["segm"]
    gt_depth = sample["depth"]
    img = sample["image"]
    print(gt_segm.shape, gt_depth.shape, img.shape)

    img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float() 
    if HAS_CUDA:
        img_var = img_var.cuda().float()
    
    depth, segm = model(img_var)

    segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                      img.shape[:2][::-1],
                      interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                       img.shape[:2][::-1],
                       interpolation=cv2.INTER_CUBIC)
    segm = segm.argmax(axis=2) + 1
    depth = np.abs(depth)

plt.figure(figsize=(18, 12))
plt.subplot(151)
plt.imshow(img)
plt.title('orig img')
plt.axis('off')
plt.subplot(152)
plt.imshow(gt_segm + 1)
plt.title('gt segm')
plt.axis('off')
plt.subplot(153)
plt.imshow(segm)
plt.title('pred segm')
plt.axis('off')
plt.subplot(154)
plt.imshow(gt_depth / DEPTH_COEFF, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
plt.title('gt depth')
plt.axis('off')
plt.subplot(155)
plt.imshow(depth, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
plt.title('pred depth')
plt.axis('off')
plt.show()