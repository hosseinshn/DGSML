from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.init as init
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from utilspv2 import *

def resnet(num_classes, pretrained=True):
    model = resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model

class DGresnet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DGresnet, self).__init__()
        self.base_model = resnet(num_classes=num_classes, pretrained=pretrained)
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        output_class = self.base_model.fc(x)
        return output_class
        
    def features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()    
            

class Classifier_homo(nn.Module):
    def __init__(self, num_classes):
        super(Classifier_homo, self).__init__()
        self.CLS = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, num_classes))
    def forward(self, x, param = None):
        if param == None: 
            x = self.CLS(x)
        else: 
            x = F.relu(x)
            x = F.linear(x, param['CLS.1.weight'], param['CLS.1.bias'])
        return x
    
    
def cloned_state_dict(Model):
    cloned_state_dict = {
        key: val.clone()
        for key, val in Model.state_dict().items()
    }
    return cloned_state_dict     