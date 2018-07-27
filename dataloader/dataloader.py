import logging
import os

import torch.utils.data as td
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import utils

logger = logging.getLogger('trendage')
import cv2
import dataset
import dataloader
import torch
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision import transforms

import numpy as np


class HDDLoader(td.Dataset):
    def __init__(self, dataset, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.len = len(self.data)
        self.dataset = dataset

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        Replacing this with a more efficient implementation selection; removing c
        :param index:
        :return:
        """

        totensor = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor()])

        assert (index < len(self.data))
        assert (index < self.len)
        images = self.data[index]
        #print(images)
        img = cv2.imread(os.path.join(self.dataset.directory, images))

        target = self.labels[index]

        scale = np.array(img.shape) / 224

        #print(img.shape, scale)
        img = cv2.resize(img, (224, 224))

        #print(target)

        target[0] = int(target[0] / scale[0])
        target[1] = int(target[1] / scale[1])
        target[2] = int(target[2] / scale[0])
        target[3] = int(target[3] / scale[1])

        #print(target)

        if self.transform is not None:
            seq_det = self.transform.to_deterministic()  # call this for each batch again, NOT only once at the start

            keypoints_on_images = []
            keypoints = []
            keypoints.append(ia.Keypoint(x=target[0], y=target[1]))
            keypoints.append(ia.Keypoint(x=target[2], y=target[3]))

            keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=np.asarray(img).shape[:-1]))

            # augment keypoints and images
            img = seq_det.augment_image(np.asarray(img))
            after_aug = []

            target = seq_det.augment_keypoints(keypoints_on_images)
            for point in target[0].keypoints:
                #print(point)
                x_new, y_new = point.x, point.y
                after_aug.append(point.x)
                after_aug.append(point.y)
            target = after_aug
            #print(after_aug)
        newImg = Image.fromarray(img)
        return totensor(newImg), np.array(target), index


'''
dataset = dataset.Trendage("/Users/nomanshafqat/Desktop/newdata/")

train_loader = dataloader.HDDLoader(dataset, dataset.train_data, dataset.train_labels, dataset.transform)
val_loader = dataloader.HDDLoader(dataset, dataset.val_data, dataset.val_labels, dataset.transform)

kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}

train_iterator = torch.utils.data.DataLoader(train_loader,
                                             batch_size=2, shuffle=True, **kwargs)

val_iterator = torch.utils.data.DataLoader(val_loader,
                                           batch_size=2, shuffle=True, **kwargs)
a = train_iterator.__iter__()

print(next(a))
'''
