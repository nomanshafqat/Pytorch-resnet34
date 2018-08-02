import argparse
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

import  cv2
import numpy as np
import models

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

        regr,clas = model(input1)

        regr=np.array(regr[0]).astype(int)
        clas=clas.numpy()[0]
        #print(regr)
        clas=np.argmax(clas,-1)
        draw=input
        if clas==1:
            draw=cv2.rectangle(cv2.resize(input,(224,224)),(regr[0],regr[1]),(regr[2],regr[3]),color=(0,255,0),thickness=3)

        cv2.imwrite(os.path.join("res",imgname),draw)
