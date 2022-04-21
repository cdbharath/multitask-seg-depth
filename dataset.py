from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
import numpy as np
import glob

class NYUDDataset(Dataset):
    """
    The dataset is downloaded from http://dl.caffe.berkeleyvision.org/nyud.tar.gz
    """
    def __init__(self, img_paths, seg_paths, depth_paths, transform=None):
        super().__init__()

        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.depth_paths = depth_paths
        self.transform = transform
        self.mask_names = ("depth", "segm")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample = {"image": np.array(Image.open(self.img_paths[idx])),
                  "segm": np.array(scipy.io.loadmat(self.seg_paths[idx])["segmentation"]),
                  "depth": np.array(Image.open(self.depth_paths[idx])),
                  "names":self.mask_names}
        if self.transform:
            sample = self.transform(sample)
            # if "names" in sample:
            #     del sample["names"]
        return sample

class CityscapesDataset(Dataset):
    """
    The dataset should be downloaded from
    1. https://www.cityscapes-dataset.com/file-handling/?packageID=1
    2. https://www.cityscapes-dataset.com/file-handling/?packageID=3
    3. https://www.cityscapes-dataset.com/file-handling/?packageID=7
    
    and placed in a directory named cityscapes 
    """
    def __init__(self, img_paths, seg_paths, ins_paths, depth_paths, transform=None):
        super().__init__()

        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.ins_paths = ins_paths
        self.depth_paths = depth_paths
        self.transform = transform
        self.mask_names = ("depth", "segm", "ins")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample = {"image": np.array(Image.open(self.img_paths[idx])),
                  "segm": np.array(Image.open(self.seg_paths[idx])),
                  "ins": np.array(Image.open(self.ins_paths[idx])),
                  "depth": np.array(Image.open(self.depth_paths[idx])),
                  "names":self.mask_names}
        if self.transform:
            sample = self.transform(sample)
            # if "names" in sample:
            #     del sample["names"]
        return sample

if __name__ == "__main__":

    # For NYUD dataset
    # img_paths = sorted(glob.glob("./nyud/data/images/*"))
    # seg_paths = sorted(glob.glob("./nyud/segmentation/*"))
    # depth_paths = sorted(glob.glob("./nyud/data/depth/*"))

    # dataset = NYUDDataset(img_paths, seg_paths, depth_paths)
    # sample = dataset[5]

    # f, ax = plt.subplots(1,3)
    # ax[0].imshow(sample["image"])
    # ax[1].imshow(sample["segm"])
    # ax[2].imshow(sample["depth"]) 
    # plt.show() 

    # For Cityscapes dataset

    img_paths = sorted(glob.glob("./cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*"))
    seg_paths = sorted(glob.glob("./cityscapes/gtFine_trainvaltest/gtFine/train/*/*labelIds.png"))
    ins_paths = sorted(glob.glob("./cityscapes/gtFine_trainvaltest/gtFine/train/*/*instanceIds.png"))
    depth_paths = sorted(glob.glob("./cityscapes/disparity_trainvaltest/disparity/train/*/*"))

    dataset = CityscapesDataset(img_paths, seg_paths, ins_paths, depth_paths)
    sample = dataset[0]

    f, ax = plt.subplots(1,4)
    ax[0].imshow(sample["image"])
    ax[1].imshow(sample["segm"])
    ax[2].imshow(sample["ins"]) 
    ax[3].imshow(sample["depth"]) 
    plt.show() 
