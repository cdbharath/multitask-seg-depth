from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io
import numpy as np

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
    pass

if __name__ == "__main__":
    img_paths = sorted(glob.glob("./nyud/data/images/*"))
    seg_paths = sorted(glob.glob("./nyud/segmentation/*"))
    depth_paths = sorted(glob.glob("./nyud/data/depth/*"))

    dataset = NYUDDataset(img_paths, seg_paths, depth_paths)
    sample = dataset[5]

    f, ax = plt.subplots(1,3)
    ax[0].imshow(sample["image"])
    ax[1].imshow(sample["segm"])
    ax[2].imshow(sample["depth"]) 
    plt.show() 
