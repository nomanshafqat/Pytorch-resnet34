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


    def get_model(self):
        resnet34 = models.resnet34(pretrained=True)

        return resnet34


    def forward(self, x):
        #print(list(self.resnet34.children()))

        output=self.resnet34(x)

        #output=nn.Sequential(*list(self.resnet34.children())[:-1])
        #removed=list(self.resnet34.children())[:-1]
        #output = nn.Sequential(*removed)

        #model = nn.Sequential(output,nn.Linear(2048, 365))

        #print(output.shape)

        return output