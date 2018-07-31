# Trendage : Generate Good looking Outfits from a Dynamic Inventory
# Authors : Omer Javed, Khurram Javed, Hassan Mahmood, Noman Shafqat
# Corresponding Author : Khurram Javed (kjaved@ualberta.ca)

import ast
import csv
import itertools
import logging
import os
import random
from random import shuffle

import numpy as np
from torchvision import transforms
from tqdm import tqdm

import utils

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from sklearn.utils import shuffle

logger = logging.getLogger('trendage')


# filename not an image file
class Dataset:
    """
    Base class to represent a Dataset
    """

    def __init__(self, name, directory):
        self.name = name
        self.data = []
        self.bbox = []
        self.directory = directory


class Trendage(Dataset):
    """
    """

    def __init__(self, directory,ne=None):

        super().__init__("ddwd",directory)
        self.data = []
        self.bbox = []
        self.labels=[]
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor()])

        self.data=[name for name in os.listdir(directory) if ".jpg" in name[len(name)-4:]]
        shuffle(self.data)

        for i,name in enumerate(self.data):
            #print(name)
            str=open(os.path.join(directory,name)[:-3]+"txt").readlines()
            bbox=str[0].split(",")
            box=[int (a) for a in bbox]
            #print(len(bbox))
            #print(box)
            self.bbox.append(box[:4])
            #print(box[:4],)
            if len(bbox)==5:
                self.labels.append(0)
            else:
                self.labels.append(1)

        print(len(self.bbox),len(self.data))



        p = np.random.permutation(len(self.data))

        self.labels=np.array(self.labels)[p]
        self.bbox=np.array(self.bbox)[p]
        self.data=np.array(self.data)[p]




        beg_index = 0
        mid_index = int(len(self.data) * 0.9)
        end_index = len(self.data)
        self.train_data = self.data[beg_index:mid_index]
        self.train_labels = self.labels[beg_index:mid_index]
        self.train_bboxes = self.bbox[beg_index:mid_index]

        self.val_data = self.data[mid_index:end_index]
        self.val_labels = self.labels[mid_index:end_index]
        self.val_bboxes = self.bbox[beg_index:mid_index]

        sometimes = lambda aug: iaa.Sometimes(0.8, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                #iaa.Flipud(0.2),  # vertically flip 20% of all images

                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.15, 0.15),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                ))

            ],
            random_order=True
        )

        self.transform = seq


if __name__ == "__main__":
    Trendage("/Users/nomanshafqat/Desktop/newdata/")


