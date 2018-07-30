import torchvision.models as models
from torch import nn



class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet34.fc = nn.Linear(512, 4)

        #self.conv2d = nn.Conv2d(1000, 500, kernel_size=1)
        #self.fc=nn.Linear(512, 4)
        return


    def forward(self, x):

        output=self.resnet34(x)

        return output