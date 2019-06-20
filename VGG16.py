import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(torch.nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        features=list(vgg16(pretrained=True).features)[:23]
        self.features=nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for i,model in enumerate(self.features):
            x=model(x)
            if i in (3,8,15,22):
                results.append(x)

        vgg_output=namedtuple('vggoutput', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_output(*results)


#%%
from collections import namedtuple
vgg_output=namedtuple('vggoutput', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
print(1)