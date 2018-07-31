import torchvision.models as models
from torch import nn



class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)

        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-1])

        self.regr = nn.Linear(512, 4)
        self.clas= nn.Linear(512, 2)
        print(self.resnet34)
        return


    def forward(self, x):

        avgpool=self.resnet34(x)
        avgpool = avgpool.reshape(avgpool.size(0), -1)

        #print(avgpool.shape)
        clas=self.clas(avgpool)
        regr=self.regr(avgpool)

        return regr,clas