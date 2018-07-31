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
    def __init__(self, dataset, data, bbox,labels, transform):
        self.data = data
        self.labels = labels
        self.bbox=bbox
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
             transforms.ToTensor()
             ])

        assert (index < len(self.data))
        assert (index < self.len)
        images = self.data[index]
        # print(images)
        img = cv2.imread(os.path.join(self.dataset.directory, images))

        target = self.bbox[index]

        scale = np.array(img.shape) / 224

        # img = cv2.rectangle(img, (target[0]-10, target[1]-10), (target[2]+10, target[3]+10),
        #                     color=(255, 255, 0), thickness=10)

        # cv2.imwrite(os.path.join("res", str(index)+".jpg"), draw)

        # print(img.shape, scale)
        img = cv2.resize(img, (224, 224))

        # print(target)

        target[0] = int(target[0] / scale[1] - 5)
        target[1] = int(target[1] / scale[0] - 5)
        target[2] = int(target[2] / scale[1] + 5)
        target[3] = int(target[3] / scale[0] + 5)

        # print(target)
        t = target
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
                # print(point)
                x_new, y_new = point.x, point.y
                after_aug.append(point.x)
                after_aug.append(point.y)
            target = after_aug
            # print(after_aug)
        newImg = Image.fromarray(img)
        reg_targets = np.float32(np.array(target))

        b=self.labels[index]

        #a = np.array(self.labels[index])
        #b = np.zeros((a.size, 2))
        #b[np.arange(a.size), a] = 1

        #print("B=",b,self.labels[index])

        #print(targets)
        ##draw = cv2.rectangle(cv2.resize(np.array(newImg), (224, 224)), (t[1], t[0]), (t[3], t[2]), color=(0, 0, 0),
        #                     thickness=6)

        #draw = cv2.rectangle(cv2.resize(np.array(draw), (224, 224)), (targets[0], targets[1]), (targets[2], targets[3]),
        #                     color=(0, 255, 0), thickness=3)

        #cv2.imwrite(os.path.join("res", str(index) + ".jpg"), draw)
        #print(reg_targets)

        return totensor(newImg), reg_targets,b ,index


if __name__ == "__main__":

    dataset = dataset.Trendage("/Users/nomanshafqat/Desktop/newdata/","/Users/nomanshafqat/Desktop/negatives")

    train_loader = dataloader.HDDLoader(dataset, dataset.train_data,dataset.bbox, dataset.train_labels, dataset.transform)
    val_loader = dataloader.HDDLoader(dataset, dataset.val_data,dataset.bbox, dataset.val_labels, dataset.transform)

    kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}

    train_iterator = torch.utils.data.DataLoader(train_loader,
                                                 batch_size=500, shuffle=True, **kwargs)

    val_iterator = torch.utils.data.DataLoader(val_loader,
                                               batch_size=2, shuffle=True, **kwargs)
    a = train_iterator.__iter__()

    next(a)

