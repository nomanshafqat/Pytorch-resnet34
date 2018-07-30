import argparse

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from tqdm import tqdm
import  cv2
import numpy as np
import utils
import dataset
import models
import dataloader
import trainer

parser = argparse.ArgumentParser(description='Pen refinement')

parser.add_argument('--dir', default="", help="Directory containing data")

parser.add_argument('--cuda', action='store_true', default=False, help=' CUDA training')

parser.add_argument('--ckpt', default="", help='without CUDA training')

args = parser.parse_args()

# args.cuda = args.cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

model = models.model().eval()
print(model.state_dict().keys())



load=torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})

print(load.keys())
pretrained_dict = {k: v for k, v in load.items() if k in model.state_dict()}

model.load_state_dict(pretrained_dict)

trans1 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
with torch.no_grad():

    for imgname in [name for name in os.listdir(args.dir) if ".jpg" in name[len(name) - 4:]]:
        print((imgname))
        input1 = Image.open(os.path.join(args.dir, imgname))


        #input=torch.from_numpy(input)
        if args.cuda:
            input1 = input1.cuda()
            model.cuda()
        #print("input=",input.shape)
        #input=torch.Tensor(input).unsqueeze_(0)

        image_tensor=trans1(input1).float().unsqueeze_(0)

        input1 = Variable(image_tensor)
        #print("input=",input1.shape,input1)

        input=cv2.imread(os.path.join(args.dir, imgname))
        output = model(input1)
        output=np.array(output[0]).astype(int)
        print(output)

        draw=cv2.rectangle(cv2.resize(input,(224,224)),(output[0],output[1]),(output[2],output[3]),color=(0,255,0),thickness=3)

        cv2.imwrite(os.path.join("res",imgname),draw)
