import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from utils import Normalise, RandomCrop, ToTensor, RandomMirror, Resize, ToOnehot, Crop, DiscriminativeLoss
from dataset import CityscapesDataset
from torch.utils.data import DataLoader
from mnet.model import MNET
from utils import InvHuberLoss
from utils import AverageMeter
from utils import MeanIoU, RMSE
from tqdm import tqdm

cwd = os.path.dirname(os.path.abspath(__file__))

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(cwd, "logs", "run_" + timestr) 
os.makedirs(log_dir)

torch.autograd.detect_anomaly()

num_classes = (1, 6)
num_instances = 16
tasks = [False, True, False]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda: " + str(device))
crop_size = 400
img_scale = 1.0 / 255
depth_scale = 100.0

img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])
transform_train = transforms.Compose([RandomMirror(),
                                      # RandomCrop(crop_size=crop_size),
                                      # Crop(0, 0.85, 0, 1),
                                      Resize((224, 244)),
                                      Normalise(scale=img_scale, mean=img_mean.reshape((1,1,3)), std=img_std.reshape(((1,1,3))), depth_scale=depth_scale),
                                      ToTensor(),
                                      ToOnehot(num_instances=num_instances)])
transform_valid = transforms.Compose([Resize((224, 244)),
                                      Normalise(scale=img_scale, mean=img_mean.reshape((1,1,3)), std=img_std.reshape(((1,1,3))), depth_scale=depth_scale),
                                      ToTensor()])

train_batch_size = 32
valid_batch_size = 32

train_img_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*")))
train_seg_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/gtFine_trainvaltest/gtFine/train/*/*labelIds.png")))
train_ins_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/gtFine_trainvaltest/gtFine/train/*/*instanceIds.png")))
train_depth_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/disparity_trainvaltest/disparity/train/*/*")))
val_img_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/*/*")))
val_seg_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/gtFine_trainvaltest/gtFine/val/*/*labelIds.png")))
val_ins_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/gtFine_trainvaltest/gtFine/val/*/*instanceIds.png"))) 
val_depth_paths = sorted(glob.glob(os.path.join(cwd, "cityscapes/disparity_trainvaltest/disparity/val/*/*")))

print("[INFO]: Loading data")
trainloader = DataLoader(CityscapesDataset(train_img_paths, train_seg_paths, train_ins_paths, train_depth_paths, transform=transform_train),
                         batch_size=train_batch_size,
                         shuffle=True, num_workers=8,
                         drop_last=True)
valloader = DataLoader(CityscapesDataset(val_img_paths, val_seg_paths, val_ins_paths, val_depth_paths, transform=transform_valid),
                       batch_size=valid_batch_size,
                       shuffle=False, num_workers=8,
                       drop_last=False)

print("[INFO]: Loading model")

MNET = MNET(tasks=[False, True, False], num_classes=num_classes[1], num_instances=None)

# Load mobile net pretrained weight for training
ckpt = torch.load(os.path.join(cwd, "weights/mobilenetv2-pretrained.pth"), map_location=device)
MNET.enc.load_state_dict(ckpt)

# Load both encoder and decoder with pretrained weights from the reference paper
# ckpt = torch.load(os.path.join(cwd, 'weights/ExpNYUD_joint.ckpt'), map_location=device)
# MNET.enc.load_state_dict(ckpt["state_dict"], strict=False)
# MNET.dec.load_state_dict(ckpt["state_dict"], strict=False)

MNET.to(device)
print("[INFO]: Model has {} parameters".format(sum([p.numel() for p in MNET.parameters()])))
print("[INFO]: Model and weights loaded successfully")

# for param in MNET.enc.parameters():
#     param.requires_grad=False

ignore_index = 255
ignore_depth = -1

crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
# crit_depth = InvHuberLoss(ignore_index=ignore_depth).to(device)

crit_insegm = DiscriminativeLoss(delta_var=0.5,
                                    delta_dist=1.5,
                                    norm=2,
                                    usegpu=True).to(device)
# crit_depth = nn.MSELoss().to(device)
# crit_depth = nn.L1Loss().to(device)

lr_encoder = 1e-3
lr_decoder = 1e-3
momentum_encoder = 0.8
momentum_decoder = 0.8
weight_decay_encoder = 1e-5
weight_decay_decoder = 1e-5

n_epochs = 500

optims = [torch.optim.SGD(MNET.enc.parameters(), lr=lr_encoder, momentum=momentum_encoder, weight_decay=weight_decay_encoder),
          torch.optim.SGD(MNET.dec.parameters(), lr=lr_decoder, momentum=momentum_decoder, weight_decay=weight_decay_decoder)]

opt_scheds = []
for opt in optims:
    opt_scheds.append(torch.optim.lr_scheduler.MultiStepLR(opt, np.arange(0, n_epochs, 100), gamma=0.1))


def train(model, opts, crits, dataloader, loss_coeffs=(1.0,), grad_norm=0.0):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        image = sample["image"].float().to(device)
        targets = [sample[k].to(device) for k in dataloader.dataset.mask_names]        
        output = model(image)

        for out, target, crit, loss_coeff, mask, task in zip(output, targets, crits, loss_coeffs, dataloader.dataset.mask_names, tasks):
            if mask != "ins":
                target_size = target.size()[1:]
            else:
                target_size = target.size()[2:]

            if not task:
                continue

            loss += loss_coeff * crit(F.interpolate(out, target_size, mode="bilinear", align_corners=False).squeeze(dim=1),
                                    target.squeeze(dim=1))


            # Uncomment while using mean squared error
            # if mask == "ins":
            #     continue
            # elif mask == "depth":
            #     loss += loss_coeff * torch.sqrt(crit(F.interpolate(out, target_size, mode="bilinear", align_corners=False).squeeze(dim=1).float(),
            #                             target.squeeze(dim=1).float()))
            # else:
            #     loss += loss_coeff * crit(F.interpolate(out, target_size, mode="bilinear", align_corners=False).squeeze(dim=1),
            #                             target.squeeze(dim=1))

            # Uncomment while not using instance head
            # if mask == "ins":
            #     # print(crit(F.interpolate(out, target_size, mode="bilinear", align_corners=False),
            #     #                     target))
            #     continue
            # loss += loss_coeff * crit(F.interpolate(out, target_size, mode="bilinear", align_corners=False).squeeze(dim=1),
            #                         target.squeeze(dim=1))

        for opt in opts:
            opt.zero_grad()
        loss.backward()
        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        for opt in opts:
            opt.step()
        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )

    return loss_meter.avg

def validate(model, metrics, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model.eval()
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            image = sample["image"].float().to(device)
            targets = [sample[k].to(device) for k in dataloader.dataset.mask_names]

            targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]

            # Forward
            outputs = model(image)

            # Backward
            for out, target, metric, task in zip(outputs, targets, metrics, tasks):
                if not task:
                    continue
                metric.update(
                    F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
                    .squeeze(dim=1)
                    .cpu()
                    .numpy(),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, val_str = get_val(metrics)
    print("Val Metrics: " + val_str)
    print("----" * 5)
    return vals


loss_accumulator = []
depth_rmse_accumulator = []
sem_meaniou_accumulator = []
disc_loss_accumulator = []

val_every = 5
loss_coeffs = (0.5, 0.5, 0.5)
print("[INFO]: Start Training")
for i in range(0, n_epochs):

    print("Epoch {:d}".format(i))
    avg_loss = train(MNET, optims, [crit_depth, crit_segm, crit_insegm], trainloader, loss_coeffs)

    print("Avg Training Loss {:.3f}".format(avg_loss))

    for sched in opt_scheds:
        sched.step()

    if i % val_every == 0:
        metrics = [RMSE(ignore_val=ignore_depth), MeanIoU(num_classes[1]), MeanIoU(num_instances)]

        with torch.no_grad():
            vals = validate(MNET, metrics, valloader)

    loss_accumulator.append(avg_loss)
    plt.figure(1)
    plt.title("Training Loss")
    plt.plot(loss_accumulator)
    plt.savefig(os.path.join(log_dir, "training_loss.png"))

    if tasks[0]: 
        depth_rmse_accumulator.append(vals[0])
        plt.figure(2)
        plt.title("RMSE Depth Estimation")
        plt.plot(depth_rmse_accumulator)
        plt.savefig(os.path.join(log_dir, "rmse_depth.png"))

    if tasks[1]:
        sem_meaniou_accumulator.append(vals[1])
        plt.figure(3)
        plt.title("Mean IOU Semantic Segmentation")
        plt.plot(sem_meaniou_accumulator)
        plt.savefig(os.path.join(log_dir, "meaniou_sem.png"))

    if tasks[2]:       
        disc_loss_accumulator.append(vals[2])

        # plt.figure(4)
        # plt.title("Discriminative Loss Instance Segmentation")
        # plt.plot(disc_loss_accumulator)
        # plt.savefig(os.path.join(log_dir, "disc_loss.png"))

    if i%50 == 0:
        print("Saving Checkpoint")
        torch.save(MNET.state_dict(), os.path.join(log_dir, "checkpoint_epoch" + str(i) + ".pth"))
