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
        self.mask_names = ("depth", "segm", "ins")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample = {"image": np.array(Image.open(self.img_paths[idx])),
                  "segm": np.array(scipy.io.loadmat(self.seg_paths[idx])["segmentation"]),
                  "depth": np.array(Image.open(self.depth_paths[idx])),
                  "ins": np.array(scipy.io.loadmat(self.seg_paths[idx])["segmentation"]),
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

        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.labels = [
            (                   "name","id", "trainId",         "category",  "catId","hasInstances","ignoreInEval",        "color"),
            (  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            (  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            (  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            (  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            (  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            (  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            (  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            (  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            (  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            (  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            (  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            (  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            (  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            (  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            (  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            (  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            (  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            (  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            (  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            (  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            (  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            (  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            (  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            (  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            (  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            (  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            (  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            (  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            (  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            (  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            (  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        disparity = np.array(Image.open(self.depth_paths[idx])).astype(np.float32)
        disparity[disparity > 0] = (disparity[disparity > 0] - 1)/256. 
        disparity[disparity > 0] = (0.209313*2262.52)/disparity[disparity > 0]
        disparity[disparity == 0] = 500.

        # TODO remove hardcode
        semantic = np.array(Image.open(self.seg_paths[idx]))

        for i in range(1, 36):
            semantic[semantic == self.labels[i][1]] = self.labels[i][4]
        
        ins = np.array(Image.open(self.ins_paths[idx])).astype(np.float32)
        ins[ins//1000 != 26] = 16
        ins[ins//1000 == 26] = ins[ins//1000 == 26]%1000
        ins[ins >= 16] = 16
        
        sample = {"image": np.array(Image.open(self.img_paths[idx])),
                  "segm": semantic,
                  "ins": ins,
                  "depth": disparity,
                  "names":self.mask_names}
        if self.transform:
            sample = self.transform(sample)
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
